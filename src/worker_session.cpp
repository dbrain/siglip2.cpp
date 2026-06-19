// worker_session.cpp — see worker_session.h for design.
//
// Child-side note: this is the GPU process. ggml-cuda primary context,
// VisionEncoder + TextEncoder weights, the megakernel cache — all live
// here. SIGKILL from the parent's shutdown() reclaims every byte; the
// next ensure_loaded() respawns from scratch via fork+execv.

#include "worker_session.h"

#include "siglip2_vision.h"
#include "siglip2_text.h"
#include "siglip2_tokenizer.h"
#include "siglip2_score.h"
#include "siglip2_preproc.h"
#include "cuda/siglip2_megakernel.h"
#include "stb_image.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#if defined(__linux__)
#  include <sys/prctl.h>
#endif

using nlohmann::json;

namespace siglip2 {

// ────────────────────────────── helpers ──────────────────────────────

static void l2_normalize(std::vector<float> & v) {
    double s = 0.0;
    for (float x : v) s += (double)x * x;
    const float inv = (float)(1.0 / (std::sqrt(s) + 1e-12));
    for (auto & x : v) x *= inv;
}

static Pooling parse_pooling(const std::string & s, std::string & err) {
    if (s.empty() || s == "pooler" || s == "probe") return Pooling::PROBE;
    if (s == "mean") return Pooling::MEAN;
    err = "pooling must be one of: pooler, probe, mean (got '" + s + "')";
    return Pooling::PROBE;
}

// Persistent single-thread worker mirroring siglip2_server.cpp's; lifted
// here so the child process keeps the same parallel-streams pattern for
// /v1/classify (vision on this thread, text on the worker thread, both
// on private CUDA streams).
struct PersistentWorker {
    std::mutex                m;
    std::condition_variable   cv_req;
    std::condition_variable   cv_done;
    std::function<void()>     task;
    bool                      pending  = false;
    bool                      shutdown = false;
    bool                      started  = false;
    std::thread               th;

    void start() {
        if (started) return;
        started = true;
        th = std::thread([this] { run(); });
    }
    ~PersistentWorker() {
        if (!started) return;
        {
            std::lock_guard<std::mutex> lk(m);
            shutdown = true;
        }
        cv_req.notify_all();
        if (th.joinable()) th.join();
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(m);
            task = std::move(f);
            pending = true;
        }
        cv_req.notify_one();
    }
    void wait_done() {
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [this] { return !pending; });
    }
private:
    void run() {
        std::unique_lock<std::mutex> lk(m);
        while (true) {
            cv_req.wait(lk, [this] { return pending || shutdown; });
            if (shutdown) return;
            auto f = std::move(task);
            task = nullptr;
            lk.unlock();
            f();
            lk.lock();
            pending = false;
            cv_done.notify_one();
        }
    }
};

// Worker-side state. Subset of siglip2_server.cpp's ServerState — no
// HTTP fields, but the same encoder + parallel-streams scaffolding.
struct WorkerChildState {
    WorkerLoadConfig                cfg;
    std::mutex                      vision_mutex;
    std::mutex                      text_mutex;
    std::mutex                      text_worker_mutex;
    PersistentWorker                text_worker;
    std::unique_ptr<VisionEncoder>  vision;
    std::unique_ptr<TextEncoder>    text;
    std::unique_ptr<Tokenizer>      tokenizer;
    ScoreParams                     score_params;
    bool                            loaded     = false;
    bool                            dual_tower = false;

    bool load(std::string & err) {
        std::scoped_lock lock(vision_mutex, text_mutex);
        if (loaded) return true;
        if (cfg.model_path.empty()) {
            err = "model path required";
            return false;
        }
        const bool need_text = !cfg.vision_only;
        if (need_text && cfg.tokenizer_path.empty()) {
            err = "tokenizer path required (or pass --vision-only)";
            return false;
        }
        dual_tower = !cfg.vision_only && !cfg.text_only;
        if (!cfg.text_only) {
            vision = std::make_unique<VisionEncoder>();
            if (!vision->load(cfg.model_path, /*private_backend=*/dual_tower)) {
                err = "vision load: " + vision->last_error();
                return false;
            }
        }
        if (need_text) {
            text      = std::make_unique<TextEncoder>();
            tokenizer = std::make_unique<Tokenizer>();
            if (!text->load(cfg.model_path, /*private_backend=*/dual_tower)) {
                err = "text load: " + text->last_error();
                return false;
            }
            if (!tokenizer->load(cfg.tokenizer_path)) {
                err = "tokenizer load: " + tokenizer->last_error();
                return false;
            }
        }
        if (dual_tower) {
            if (!read_score_params(cfg.model_path, score_params, err)) {
                return false;
            }
        }
        loaded = true;
        if (dual_tower) text_worker.start();
        const char * mode = cfg.vision_only ? " (vision-only)" :
                            cfg.text_only   ? " (text-only)"   : "";
        fprintf(stderr, "[siglip2-worker] model loaded: %s%s%s\n",
                cfg.model_path.c_str(), mode,
                dual_tower ? " [private-streams]" : "");
        return true;
    }
};

// Decode an image (jpeg/png/...) from a memory buffer to row-major HWC RGB uint8.
static bool decode_image(const std::string & bytes, std::vector<uint8_t> & rgb,
                         int & h, int & w, std::string & err) {
    int channels = 0;
    uint8_t * px = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc *>(bytes.data()), (int)bytes.size(),
        &w, &h, &channels, /*req_comp=*/3);
    if (!px) {
        err = std::string("stbi_load_from_memory failed: ")
            + (stbi_failure_reason() ? stbi_failure_reason() : "?");
        return false;
    }
    rgb.assign(px, px + (size_t)h * w * 3);
    stbi_image_free(px);
    return true;
}

static bool encode_image_locked(
    WorkerChildState &  st,
    const std::string & bytes,
    int                 max_num_patches,
    Pooling             pooling,
    std::vector<float> & out_emb,
    std::string &       err) {
    std::vector<uint8_t> rgb;
    int h = 0, w = 0;
    if (!decode_image(bytes, rgb, h, w, err)) return false;

    PreprocResult pp;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std_v[3] = {0.5f, 0.5f, 0.5f};
    if (!preprocess_image_rgb(rgb.data(), h, w, max_num_patches,
                              st.vision->config().patch_size,
                              1.0f / 255.0f, mean, std_v, pp, err)) {
        return false;
    }
    if (!st.vision->encode(pp.pixel_values.data(), pp.n_patches_h, pp.n_patches_w,
                           pooling, out_emb)) {
        err = "vision encode: " + st.vision->last_error();
        return false;
    }
    return true;
}

static bool encode_text_batch_locked(
    WorkerChildState &                st,
    const std::vector<std::string> &  prompts,
    std::vector<std::vector<float>> & out_embs,
    std::string &                     err) {
    out_embs.clear();
    if (prompts.empty()) return true;

    const int n_batch = (int)prompts.size();
    const int max_pos = st.text->config().max_position_embeddings;
    const int proj    = st.text->config().projection_size;

    std::vector<int32_t> ids_flat((size_t)max_pos * (size_t)n_batch);
    for (int b = 0; b < n_batch; ++b) {
        std::vector<int32_t> ids, mask;
        if (!st.tokenizer->encode(prompts[b], max_pos, ids, mask)) {
            err = "tokenize: " + st.tokenizer->last_error();
            return false;
        }
        std::memcpy(ids_flat.data() + (size_t)b * (size_t)max_pos,
                    ids.data(), sizeof(int32_t) * (size_t)max_pos);
    }

    std::vector<float> flat;
    if (!st.text->encode_batch(ids_flat.data(), max_pos, n_batch,
                               /*attention_mask=*/nullptr, flat)) {
        err = "text encode_batch: " + st.text->last_error();
        return false;
    }

    out_embs.assign(n_batch, std::vector<float>((size_t)proj));
    for (int b = 0; b < n_batch; ++b) {
        std::memcpy(out_embs[b].data(),
                    flat.data() + (size_t)b * (size_t)proj,
                    sizeof(float) * (size_t)proj);
    }
    return true;
}

// ──────────────────────── WorkerSession (parent) ──────────────────────

WorkerSession::WorkerSession(const char * argv0,
                             std::vector<std::string> extra_argv)
    : argv0_(argv0 ? argv0 : ""), extra_argv_(std::move(extra_argv)) {}

WorkerSession::~WorkerSession() {
    shutdown();
}

void WorkerSession::kill_worker_locked() {
    if (pid_ > 0) {
        ::kill(pid_, SIGKILL);
        int wstat = 0;
        ::waitpid(pid_, &wstat, 0);
        fprintf(stderr, "siglip2-session: killed worker pid=%d (wstat=0x%x)\n",
                (int) pid_, wstat);
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    pid_ = -1;
    loaded_ok_  = false;
    hidden_size_ = 0;
    max_pos_    = 0;
    // Keep loaded_cfg_ so request methods can self-heal-respawn after a
    // racing /unload. ensure_loaded()'s "already loaded?" check still
    // requires loaded_ok_=true, so a stale cfg doesn't fool it.
}

void WorkerSession::shutdown() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    kill_worker_locked();
}

bool WorkerSession::send_load_req_locked(const WorkerLoadConfig & cfg) {
    json req = {
        {"model_path",              cfg.model_path},
        {"tokenizer_path",          cfg.tokenizer_path},
        {"default_max_num_patches", cfg.default_max_num_patches},
        {"vision_only",             cfg.vision_only},
        {"text_only",               cfg.text_only},
        {"lazy_load",               cfg.lazy_load},
    };
    IpcError e = send_frame(fd_, WorkerFrame::LOAD_REQ, 0, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("LOAD_REQ send: ") + ipc_error_str(e);
        return false;
    }
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("LOAD_RESP recv: ") + ipc_error_str(e);
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::LOAD_RESP)) {
        last_error_ = std::string("expected LOAD_RESP, got 0x")
                    + std::to_string(hdr.type);
        return false;
    }
    json resp;
    try {
        resp = json::parse(std::string(payload.begin(), payload.end()));
    } catch (const std::exception & ex) {
        last_error_ = std::string("LOAD_RESP parse: ") + ex.what();
        return false;
    }
    if (!resp.value("ok", false)) {
        last_error_ = std::string("worker load failed: ")
                    + resp.value("error", std::string{"(no msg)"});
        return false;
    }
    hidden_size_ = resp.value("hidden_size", 0);
    max_pos_     = resp.value("max_position_embeddings", 0);
    return true;
}

bool WorkerSession::ensure_loaded_locked(const WorkerLoadConfig & cfg,
                                         const std::string & want_gpu) {
    const std::string gpu = want_gpu.empty() ? default_gpu_ : want_gpu;
    if (pid_ > 0 && loaded_ok_
        && worker_gpu_                == gpu
        && loaded_cfg_.model_path     == cfg.model_path
        && loaded_cfg_.tokenizer_path == cfg.tokenizer_path
        && loaded_cfg_.vision_only    == cfg.vision_only
        && loaded_cfg_.text_only      == cfg.text_only) {
        return true;
    }

    if (pid_ > 0) {
        if (worker_gpu_ != gpu)
            fprintf(stderr, "siglip2-session: relocating worker '%s' -> '%s'\n",
                    worker_gpu_.c_str(), gpu.c_str());
        kill_worker_locked();
    }

    pid_t child = spawn_worker(argv0_.c_str(), extra_argv_, &fd_, gpu);
    if (child < 0) {
        last_error_ = "spawn_worker failed";
        return false;
    }
    pid_ = child;
    worker_gpu_ = gpu;

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    IpcError e = recv_frame(fd_, &hdr, &payload);
    if (e != IpcError::OK || hdr.type != static_cast<uint32_t>(WorkerFrame::HELLO)) {
        last_error_ = std::string("worker HELLO failed: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    fprintf(stderr, "siglip2-session: child pid=%d HELLO: %.*s\n",
            (int) pid_, (int) payload.size(), (const char *) payload.data());

    if (!send_load_req_locked(cfg)) {
        kill_worker_locked();
        return false;
    }
    loaded_cfg_ = cfg;
    loaded_ok_  = true;
    return true;
}

bool WorkerSession::ensure_loaded(const WorkerLoadConfig & cfg) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    return ensure_loaded_locked(cfg, std::string());
}

// Helpers for round-tripping a request: send REQ, recv RESP. On any
// transport error the worker is killed (parent-side respawn on next
// ensure_loaded()).
namespace {

bool round_trip(int fd,
                WorkerFrame req_type,
                uint32_t req_id,
                const std::vector<uint8_t> & req_payload,
                WorkerFrame want_resp,
                FrameHeader & out_hdr,
                std::vector<uint8_t> & out_payload,
                std::string & err) {
    IpcError e = send_frame(fd, req_type, req_id, req_payload);
    if (e != IpcError::OK) {
        err = std::string("send: ") + ipc_error_str(e);
        return false;
    }
    e = recv_frame(fd, &out_hdr, &out_payload);
    if (e != IpcError::OK) {
        err = std::string("recv: ") + ipc_error_str(e);
        return false;
    }
    if (out_hdr.type == static_cast<uint32_t>(WorkerFrame::ERR)) {
        try {
            json j = json::parse(std::string(out_payload.begin(), out_payload.end()));
            err = j.value("error", std::string{"(no msg)"});
        } catch (...) {
            err = "worker reported error (unparseable)";
        }
        return false;
    }
    if (out_hdr.type != static_cast<uint32_t>(want_resp)) {
        err = std::string("unexpected resp 0x") + std::to_string(out_hdr.type);
        return false;
    }
    return true;
}

} // namespace

bool WorkerSession::encode_images(
    const std::vector<std::string> &  image_bytes,
    const std::string &               pooling,
    int                               max_num_patches,
    bool                              return_last_hidden,
    std::vector<std::vector<float>> & out_embs,
    const std::string &               gpu) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    out_embs.clear();
    if (!ensure_loaded_locked(loaded_cfg_, gpu)) {
        // last_error_ is already set by ensure_loaded_locked.
        return false;
    }
    json meta = {
        {"pooling",            pooling},
        {"max_num_patches",    max_num_patches},
        {"return_last_hidden", return_last_hidden},
        {"n_images",           (int)image_bytes.size()},
    };
    auto req_payload = pack_multi_blob_payload(meta.dump(), image_bytes);

    uint32_t req_id = next_req_id_.fetch_add(1);
    FrameHeader hdr{};
    std::vector<uint8_t> resp_payload;
    std::string err;
    IpcError e = send_frame(fd_, WorkerFrame::ENCODE_IMAGES_REQ, req_id, req_payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("ENCODE_IMAGES_REQ send: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    e = recv_frame(fd_, &hdr, &resp_payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("ENCODE_IMAGES_RESP recv: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    std::string meta_str;
    std::vector<float> floats;
    if (!unpack_blob_payload(resp_payload, &meta_str, &floats)) {
        last_error_ = "ENCODE_IMAGES_RESP unpack failed";
        kill_worker_locked();
        return false;
    }
    json rmeta;
    try { rmeta = json::parse(meta_str); }
    catch (const std::exception & ex) {
        last_error_ = std::string("ENCODE_IMAGES_RESP meta parse: ") + ex.what();
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::ENCODE_IMAGES_RESP)
        || !rmeta.value("ok", false)) {
        last_error_ = rmeta.value("error", std::string{"(unknown error)"});
        return false;
    }
    int n = rmeta.value("n_images", 0);
    int H = rmeta.value("hidden_size", 0);
    if (n <= 0 || H <= 0 || (size_t)n * (size_t)H != floats.size()) {
        last_error_ = "ENCODE_IMAGES_RESP shape mismatch";
        return false;
    }
    out_embs.assign(n, std::vector<float>((size_t)H));
    for (int i = 0; i < n; ++i) {
        std::memcpy(out_embs[i].data(),
                    floats.data() + (size_t)i * (size_t)H,
                    sizeof(float) * (size_t)H);
    }
    return true;
}

bool WorkerSession::encode_texts(
    const std::vector<std::string> &  prompts,
    std::vector<std::vector<float>> & out_embs,
    const std::string &               gpu) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    out_embs.clear();
    if (!ensure_loaded_locked(loaded_cfg_, gpu)) {
        return false;
    }
    json req = { {"prompts", prompts} };
    uint32_t req_id = next_req_id_.fetch_add(1);
    FrameHeader hdr{};
    std::vector<uint8_t> resp_payload;
    std::string err;
    IpcError e = send_frame(fd_, WorkerFrame::ENCODE_TEXT_REQ, req_id, req.dump());
    if (e != IpcError::OK) {
        last_error_ = std::string("ENCODE_TEXT_REQ send: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    e = recv_frame(fd_, &hdr, &resp_payload);
    if (e != IpcError::OK) {
        last_error_ = std::string("ENCODE_TEXT_RESP recv: ") + ipc_error_str(e);
        kill_worker_locked();
        return false;
    }
    std::string meta_str;
    std::vector<float> floats;
    if (!unpack_blob_payload(resp_payload, &meta_str, &floats)) {
        last_error_ = "ENCODE_TEXT_RESP unpack failed";
        kill_worker_locked();
        return false;
    }
    json rmeta;
    try { rmeta = json::parse(meta_str); }
    catch (const std::exception & ex) {
        last_error_ = std::string("ENCODE_TEXT_RESP meta parse: ") + ex.what();
        return false;
    }
    if (hdr.type != static_cast<uint32_t>(WorkerFrame::ENCODE_TEXT_RESP)
        || !rmeta.value("ok", false)) {
        last_error_ = rmeta.value("error", std::string{"(unknown error)"});
        return false;
    }
    int n = rmeta.value("n_prompts", 0);
    int H = rmeta.value("hidden_size", 0);
    if (n <= 0 || H <= 0 || (size_t)n * (size_t)H != floats.size()) {
        last_error_ = "ENCODE_TEXT_RESP shape mismatch";
        return false;
    }
    out_embs.assign(n, std::vector<float>((size_t)H));
    for (int i = 0; i < n; ++i) {
        std::memcpy(out_embs[i].data(),
                    floats.data() + (size_t)i * (size_t)H,
                    sizeof(float) * (size_t)H);
    }
    return true;
}

static bool unpack_classify_resp(const std::vector<uint8_t> & payload,
                                 std::vector<std::vector<float>> & scores,
                                 std::vector<std::vector<float>> & logits,
                                 std::string & err) {
    json j;
    try {
        j = json::parse(std::string(payload.begin(), payload.end()));
    } catch (const std::exception & ex) {
        err = std::string("classify resp parse: ") + ex.what();
        return false;
    }
    if (!j.value("ok", false)) {
        err = j.value("error", std::string{"(unknown error)"});
        return false;
    }
    scores.clear();
    logits.clear();
    if (j.contains("scores")) {
        for (auto & row : j["scores"]) scores.push_back(row.get<std::vector<float>>());
    }
    if (j.contains("logits")) {
        for (auto & row : j["logits"]) logits.push_back(row.get<std::vector<float>>());
    }
    return true;
}

bool WorkerSession::classify(
    const std::vector<std::string> &  image_bytes,
    const std::vector<std::string> &  prompts,
    int                               max_num_patches,
    bool                              return_logits,
    std::vector<std::vector<float>> & out_scores,
    std::vector<std::vector<float>> & out_logits,
    const std::string &               gpu) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    out_scores.clear();
    out_logits.clear();
    if (!ensure_loaded_locked(loaded_cfg_, gpu)) {
        return false;
    }
    json meta = {
        {"prompts",         prompts},
        {"max_num_patches", max_num_patches},
        {"return_logits",   return_logits},
        {"n_images",        (int)image_bytes.size()},
    };
    auto req_payload = pack_multi_blob_payload(meta.dump(), image_bytes);

    uint32_t req_id = next_req_id_.fetch_add(1);
    FrameHeader hdr{};
    std::vector<uint8_t> resp_payload;
    std::string err;
    if (!round_trip(fd_, WorkerFrame::CLASSIFY_REQ, req_id, req_payload,
                    WorkerFrame::CLASSIFY_RESP, hdr, resp_payload, err)) {
        last_error_ = err;
        if (err.find("recv:") == 0 || err.find("send:") == 0) kill_worker_locked();
        return false;
    }
    if (!unpack_classify_resp(resp_payload, out_scores, out_logits, err)) {
        last_error_ = err;
        return false;
    }
    return true;
}

bool WorkerSession::classify_from_embeddings(
    const std::vector<std::vector<float>> & image_embeddings,
    const std::vector<std::string> &        prompts,
    bool                                    return_logits,
    std::vector<std::vector<float>> &       out_scores,
    std::vector<std::vector<float>> &       out_logits,
    const std::string &                     gpu) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    out_scores.clear();
    out_logits.clear();
    if (!ensure_loaded_locked(loaded_cfg_, gpu)) {
        return false;
    }
    const int n_img = (int)image_embeddings.size();
    if (n_img == 0) { last_error_ = "no image embeddings"; return false; }
    const int H = (int)image_embeddings[0].size();
    std::vector<float> flat((size_t)n_img * (size_t)H);
    for (int i = 0; i < n_img; ++i) {
        if ((int)image_embeddings[i].size() != H) {
            last_error_ = "image embedding row has mismatched dim";
            return false;
        }
        std::memcpy(flat.data() + (size_t)i * (size_t)H,
                    image_embeddings[i].data(),
                    sizeof(float) * (size_t)H);
    }

    json meta = {
        {"prompts",       prompts},
        {"return_logits", return_logits},
        {"n_images",      n_img},
        {"hidden_size",   H},
    };
    auto req_payload = pack_blob_payload(meta.dump(), flat.data(), flat.size());

    uint32_t req_id = next_req_id_.fetch_add(1);
    FrameHeader hdr{};
    std::vector<uint8_t> resp_payload;
    std::string err;
    if (!round_trip(fd_, WorkerFrame::CLASSIFY_FROM_EMB_REQ, req_id, req_payload,
                    WorkerFrame::CLASSIFY_FROM_EMB_RESP, hdr, resp_payload, err)) {
        last_error_ = err;
        if (err.find("recv:") == 0 || err.find("send:") == 0) kill_worker_locked();
        return false;
    }
    if (!unpack_classify_resp(resp_payload, out_scores, out_logits, err)) {
        last_error_ = err;
        return false;
    }
    return true;
}

// ───────────────────────── run_worker_loop (child) ─────────────────────

static void send_err(int fd, uint32_t req_id, const std::string & msg) {
    json e = { {"error", msg} };
    send_frame(fd, WorkerFrame::ERR, req_id, e.dump());
}

static void handle_load_req(WorkerChildState & st, int fd, uint32_t req_id,
                            const std::vector<uint8_t> & payload) {
    std::string err;
    bool ok = false;
    int hidden_size = 0;
    int max_pos = 0;
    try {
        json req = json::parse(std::string(payload.begin(), payload.end()));
        st.cfg.model_path              = req.value("model_path",              std::string{});
        st.cfg.tokenizer_path          = req.value("tokenizer_path",          std::string{});
        st.cfg.default_max_num_patches = req.value("default_max_num_patches", 729);
        st.cfg.vision_only             = req.value("vision_only",             false);
        st.cfg.text_only               = req.value("text_only",               false);
        st.cfg.lazy_load               = req.value("lazy_load",               false);
        if (st.cfg.lazy_load) {
            ok = true; // defer load to first encode/classify
        } else {
            ok = st.load(err);
        }
        if (st.loaded) {
            if (st.vision) hidden_size = (int)st.vision->config().hidden_size;
            else if (st.text) hidden_size = (int)st.text->config().hidden_size;
            if (st.text) max_pos = (int)st.text->config().max_position_embeddings;
        }
    } catch (const std::exception & ex) {
        err = std::string("LOAD_REQ parse: ") + ex.what();
    }
    json resp = {
        {"ok",                      ok},
        {"error",                   err},
        {"hidden_size",             hidden_size},
        {"max_position_embeddings", max_pos},
    };
    send_frame(fd, WorkerFrame::LOAD_RESP, req_id, resp.dump());
}

// Lazy-load on first encode/classify. Returns false + sets err on failure.
static bool ensure_loaded(WorkerChildState & st, std::string & err) {
    if (st.loaded) return true;
    return st.load(err);
}

static void handle_encode_images(WorkerChildState & st, int fd, uint32_t req_id,
                                 const std::vector<uint8_t> & payload) {
    std::string meta_str;
    std::vector<std::string> images;
    if (!unpack_multi_blob_payload(payload, &meta_str, &images)) {
        send_err(fd, req_id, "ENCODE_IMAGES_REQ unpack failed");
        return;
    }
    json meta;
    try { meta = json::parse(meta_str); }
    catch (const std::exception & ex) {
        send_err(fd, req_id, std::string("ENCODE_IMAGES_REQ json: ") + ex.what());
        return;
    }
    std::string err;
    if (!ensure_loaded(st, err)) {
        json m = {{"ok", false}, {"error", "lazy load: " + err}};
        send_frame(fd, WorkerFrame::ENCODE_IMAGES_RESP, req_id,
                   pack_blob_payload(m.dump(), nullptr, 0));
        return;
    }
    if (!st.vision) {
        json m = {{"ok", false}, {"error", "vision tower not loaded"}};
        send_frame(fd, WorkerFrame::ENCODE_IMAGES_RESP, req_id,
                   pack_blob_payload(m.dump(), nullptr, 0));
        return;
    }
    if (meta.value("return_last_hidden", false)) {
        json m = {{"ok", false}, {"error", "return_last_hidden not implemented"}};
        send_frame(fd, WorkerFrame::ENCODE_IMAGES_RESP, req_id,
                   pack_blob_payload(m.dump(), nullptr, 0));
        return;
    }
    Pooling pooling = parse_pooling(meta.value("pooling", std::string{}), err);
    if (!err.empty()) {
        json m = {{"ok", false}, {"error", err}};
        send_frame(fd, WorkerFrame::ENCODE_IMAGES_RESP, req_id,
                   pack_blob_payload(m.dump(), nullptr, 0));
        return;
    }
    int max_num_patches = meta.value("max_num_patches",
                                     st.cfg.default_max_num_patches);

    const int H = (int)st.vision->config().hidden_size;
    const int n = (int)images.size();
    std::vector<float> flat((size_t)n * (size_t)H);
    {
        std::lock_guard<std::mutex> lock(st.vision_mutex);
        for (int i = 0; i < n; ++i) {
            std::vector<float> emb;
            if (!encode_image_locked(st, images[i], max_num_patches, pooling, emb, err)) {
                json m = {{"ok", false}, {"error", err}};
                send_frame(fd, WorkerFrame::ENCODE_IMAGES_RESP, req_id,
                           pack_blob_payload(m.dump(), nullptr, 0));
                return;
            }
            l2_normalize(emb);
            std::memcpy(flat.data() + (size_t)i * (size_t)H,
                        emb.data(), sizeof(float) * (size_t)H);
        }
    }
    json m = {{"ok", true}, {"n_images", n}, {"hidden_size", H}};
    send_frame(fd, WorkerFrame::ENCODE_IMAGES_RESP, req_id,
               pack_blob_payload(m.dump(), flat.data(), flat.size()));
}

static void handle_encode_text(WorkerChildState & st, int fd, uint32_t req_id,
                               const std::vector<uint8_t> & payload) {
    json meta;
    try {
        meta = json::parse(std::string(payload.begin(), payload.end()));
    } catch (const std::exception & ex) {
        send_err(fd, req_id, std::string("ENCODE_TEXT_REQ json: ") + ex.what());
        return;
    }
    std::vector<std::string> prompts = meta.value("prompts", std::vector<std::string>{});
    std::string err;
    if (!ensure_loaded(st, err)) {
        json m = {{"ok", false}, {"error", "lazy load: " + err}};
        send_frame(fd, WorkerFrame::ENCODE_TEXT_RESP, req_id,
                   pack_blob_payload(m.dump(), nullptr, 0));
        return;
    }
    if (!st.text || !st.tokenizer) {
        json m = {{"ok", false}, {"error", "text tower not loaded"}};
        send_frame(fd, WorkerFrame::ENCODE_TEXT_RESP, req_id,
                   pack_blob_payload(m.dump(), nullptr, 0));
        return;
    }

    std::vector<std::vector<float>> embs;
    {
        std::lock_guard<std::mutex> lock(st.text_mutex);
        if (!encode_text_batch_locked(st, prompts, embs, err)) {
            json m = {{"ok", false}, {"error", err}};
            send_frame(fd, WorkerFrame::ENCODE_TEXT_RESP, req_id,
                       pack_blob_payload(m.dump(), nullptr, 0));
            return;
        }
    }
    for (auto & e : embs) l2_normalize(e);

    const int n = (int)embs.size();
    const int H = n > 0 ? (int)embs[0].size() : 0;
    std::vector<float> flat((size_t)n * (size_t)H);
    for (int i = 0; i < n; ++i) {
        std::memcpy(flat.data() + (size_t)i * (size_t)H,
                    embs[i].data(), sizeof(float) * (size_t)H);
    }
    json m = {{"ok", true}, {"n_prompts", n}, {"hidden_size", H}};
    send_frame(fd, WorkerFrame::ENCODE_TEXT_RESP, req_id,
               pack_blob_payload(m.dump(), flat.data(), flat.size()));
}

static void handle_classify(WorkerChildState & st, int fd, uint32_t req_id,
                            const std::vector<uint8_t> & payload) {
    std::string meta_str;
    std::vector<std::string> images;
    if (!unpack_multi_blob_payload(payload, &meta_str, &images)) {
        send_err(fd, req_id, "CLASSIFY_REQ unpack failed");
        return;
    }
    json meta;
    try { meta = json::parse(meta_str); }
    catch (const std::exception & ex) {
        send_err(fd, req_id, std::string("CLASSIFY_REQ json: ") + ex.what());
        return;
    }
    std::vector<std::string> prompts = meta.value("prompts", std::vector<std::string>{});
    int max_num_patches = meta.value("max_num_patches", st.cfg.default_max_num_patches);
    bool return_logits  = meta.value("return_logits", false);

    std::string err;
    if (!ensure_loaded(st, err)) {
        json out = {{"ok", false}, {"error", "lazy load: " + err}};
        send_frame(fd, WorkerFrame::CLASSIFY_RESP, req_id, out.dump());
        return;
    }
    if (!st.vision || !st.text || !st.tokenizer) {
        json out = {{"ok", false}, {"error", "classify requires both towers loaded"}};
        send_frame(fd, WorkerFrame::CLASSIFY_RESP, req_id, out.dump());
        return;
    }

    const int H = (int)st.vision->config().hidden_size;
    const int n_img = (int)images.size();
    const int n_txt = (int)prompts.size();
    std::vector<float> img_buf((size_t)n_img * (size_t)H);
    std::vector<float> txt_buf((size_t)n_txt * (size_t)H);

    // Parallel-streams: text on persistent worker thread, vision on this thread.
    std::lock_guard<std::mutex> worker_lock(st.text_worker_mutex);

    std::string text_err;
    std::vector<std::vector<float>> txt_embs;
    bool text_ok = false;
    st.text_worker.submit([&]() {
        std::lock_guard<std::mutex> lock(st.text_mutex);
        text_ok = encode_text_batch_locked(st, prompts, txt_embs, text_err);
    });

    bool vision_ok = true;
    {
        std::lock_guard<std::mutex> lock(st.vision_mutex);
        for (int i = 0; i < n_img; ++i) {
            std::vector<float> emb;
            if (!encode_image_locked(st, images[i], max_num_patches,
                                     Pooling::PROBE, emb, err)) {
                vision_ok = false;
                break;
            }
            std::memcpy(img_buf.data() + (size_t)i * (size_t)H,
                        emb.data(), sizeof(float) * (size_t)H);
        }
    }
    st.text_worker.wait_done();
    if (!vision_ok) {
        json out = {{"ok", false}, {"error", err}};
        send_frame(fd, WorkerFrame::CLASSIFY_RESP, req_id, out.dump());
        return;
    }
    if (!text_ok) {
        json out = {{"ok", false}, {"error", text_err}};
        send_frame(fd, WorkerFrame::CLASSIFY_RESP, req_id, out.dump());
        return;
    }
    for (int j = 0; j < n_txt; ++j) {
        std::memcpy(txt_buf.data() + (size_t)j * (size_t)H,
                    txt_embs[j].data(), sizeof(float) * (size_t)H);
    }

    std::vector<float> logits((size_t)n_img * (size_t)n_txt);
    std::vector<float> probs((size_t)n_img * (size_t)n_txt);
    score_image_text(img_buf.data(), n_img, txt_buf.data(), n_txt, H,
                     st.score_params, logits.data(), probs.data());

    std::vector<std::vector<float>> scores_out(n_img, std::vector<float>(n_txt));
    std::vector<std::vector<float>> logits_out(n_img, std::vector<float>(n_txt));
    for (int i = 0; i < n_img; ++i) {
        for (int j = 0; j < n_txt; ++j) {
            scores_out[i][j] = probs[(size_t)i * (size_t)n_txt + j];
            logits_out[i][j] = logits[(size_t)i * (size_t)n_txt + j];
        }
    }
    json out = {{"ok", true}, {"scores", scores_out}};
    if (return_logits) out["logits"] = logits_out;
    send_frame(fd, WorkerFrame::CLASSIFY_RESP, req_id, out.dump());
}

static void handle_classify_from_emb(WorkerChildState & st, int fd, uint32_t req_id,
                                     const std::vector<uint8_t> & payload) {
    std::string meta_str;
    std::vector<float> img_floats;
    if (!unpack_blob_payload(payload, &meta_str, &img_floats)) {
        send_err(fd, req_id, "CLASSIFY_FROM_EMB_REQ unpack failed");
        return;
    }
    json meta;
    try { meta = json::parse(meta_str); }
    catch (const std::exception & ex) {
        send_err(fd, req_id, std::string("CLASSIFY_FROM_EMB_REQ json: ") + ex.what());
        return;
    }
    std::vector<std::string> prompts = meta.value("prompts", std::vector<std::string>{});
    bool return_logits = meta.value("return_logits", false);
    int  n_img = meta.value("n_images", 0);
    int  H     = meta.value("hidden_size", 0);

    std::string err;
    if (!ensure_loaded(st, err)) {
        json out = {{"ok", false}, {"error", "lazy load: " + err}};
        send_frame(fd, WorkerFrame::CLASSIFY_FROM_EMB_RESP, req_id, out.dump());
        return;
    }
    if (!st.text || !st.tokenizer) {
        json out = {{"ok", false}, {"error", "text tower not loaded"}};
        send_frame(fd, WorkerFrame::CLASSIFY_FROM_EMB_RESP, req_id, out.dump());
        return;
    }
    if (!st.vision) {
        json out = {{"ok", false}, {"error", "classify requires the model loaded with both towers"}};
        send_frame(fd, WorkerFrame::CLASSIFY_FROM_EMB_RESP, req_id, out.dump());
        return;
    }
    if (n_img <= 0 || H <= 0 || (size_t)n_img * (size_t)H != img_floats.size()) {
        json out = {{"ok", false}, {"error", "image_embeddings shape mismatch"}};
        send_frame(fd, WorkerFrame::CLASSIFY_FROM_EMB_RESP, req_id, out.dump());
        return;
    }

    const int n_txt = (int)prompts.size();
    std::vector<float> txt_buf((size_t)n_txt * (size_t)H);
    {
        std::lock_guard<std::mutex> lock(st.text_mutex);
        std::vector<std::vector<float>> txt_embs;
        if (!encode_text_batch_locked(st, prompts, txt_embs, err)) {
            json out = {{"ok", false}, {"error", err}};
            send_frame(fd, WorkerFrame::CLASSIFY_FROM_EMB_RESP, req_id, out.dump());
            return;
        }
        for (int j = 0; j < n_txt; ++j) {
            std::memcpy(txt_buf.data() + (size_t)j * (size_t)H,
                        txt_embs[j].data(), sizeof(float) * (size_t)H);
        }
    }

    std::vector<float> logits((size_t)n_img * (size_t)n_txt);
    std::vector<float> probs((size_t)n_img * (size_t)n_txt);
    score_image_text(img_floats.data(), n_img, txt_buf.data(), n_txt, H,
                     st.score_params, logits.data(), probs.data());

    std::vector<std::vector<float>> scores_out(n_img, std::vector<float>(n_txt));
    std::vector<std::vector<float>> logits_out(n_img, std::vector<float>(n_txt));
    for (int i = 0; i < n_img; ++i) {
        for (int j = 0; j < n_txt; ++j) {
            scores_out[i][j] = probs[(size_t)i * (size_t)n_txt + j];
            logits_out[i][j] = logits[(size_t)i * (size_t)n_txt + j];
        }
    }
    json out = {{"ok", true}, {"scores", scores_out}};
    if (return_logits) out["logits"] = logits_out;
    send_frame(fd, WorkerFrame::CLASSIFY_FROM_EMB_RESP, req_id, out.dump());
}

int run_worker_loop(int fd) {
    setvbuf(stderr, nullptr, _IONBF, 0);

#if defined(__linux__)
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) {
        fprintf(stderr, "siglip2-worker: prctl(PR_SET_PDEATHSIG) failed: %s (continuing)\n",
                strerror(errno));
    }
#endif

    fprintf(stderr, "siglip2-worker[%d]: alive on fd=%d ppid=%d\n",
            (int) getpid(), fd, (int) getppid());

    // Megakernel hooks must be installed before any encoder load — same
    // ordering as siglip2_server.cpp's main().
    siglip2_megakernel::install();

    json hello = {
        {"pid",  (int) getpid()},
        {"role", "siglip2-worker"},
    };
    if (send_frame(fd, WorkerFrame::HELLO, 0, hello.dump()) != IpcError::OK) {
        fprintf(stderr, "siglip2-worker: HELLO send failed; bailing\n");
        return 2;
    }

    WorkerChildState st;

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        IpcError e = recv_frame(fd, &hdr, &payload);
        if (e == IpcError::EofClean) {
            fprintf(stderr, "siglip2-worker: parent EOF, exiting cleanly\n");
            return 0;
        }
        if (e != IpcError::OK) {
            fprintf(stderr, "siglip2-worker: recv_frame failed: %s\n", ipc_error_str(e));
            return 3;
        }

        switch (static_cast<WorkerFrame>(hdr.type)) {
            case WorkerFrame::SHUTDOWN:
                fprintf(stderr, "siglip2-worker: SHUTDOWN, exiting\n");
                return 0;
            case WorkerFrame::PING:
                send_frame(fd, WorkerFrame::PONG, hdr.req_id, payload);
                break;
            case WorkerFrame::LOAD_REQ:
                handle_load_req(st, fd, hdr.req_id, payload);
                break;
            case WorkerFrame::ENCODE_IMAGES_REQ:
                handle_encode_images(st, fd, hdr.req_id, payload);
                break;
            case WorkerFrame::ENCODE_TEXT_REQ:
                handle_encode_text(st, fd, hdr.req_id, payload);
                break;
            case WorkerFrame::CLASSIFY_REQ:
                handle_classify(st, fd, hdr.req_id, payload);
                break;
            case WorkerFrame::CLASSIFY_FROM_EMB_REQ:
                handle_classify_from_emb(st, fd, hdr.req_id, payload);
                break;
            default:
                fprintf(stderr, "siglip2-worker: unexpected frame 0x%x len=%u\n",
                        hdr.type, hdr.len);
                send_err(fd, hdr.req_id,
                         std::string("unexpected frame type 0x") + std::to_string(hdr.type));
                break;
        }
    }
}

} // namespace siglip2
