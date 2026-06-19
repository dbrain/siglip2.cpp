// SigLIP2 HTTP server — drop-in replacement for kobbler-vision's FastAPI service.
//
// Endpoints (all match docker/kobbler-vision/server.py request/response shape):
//   GET  /health
//   POST /unload                       (no-op; lifecycle here is in-process)
//   POST /v1/embeddings                (multipart images)
//   POST /v1/classify                  (multipart images + form prompts)
//   POST /v1/classify_from_embeddings  (JSON {image_embeddings, prompts})
//   POST /v1/text_embeddings           (form prompts)

#include "siglip2_vision.h"
#include "siglip2_text.h"
#include "siglip2_tokenizer.h"
#include "siglip2_score.h"
#include "siglip2_preproc.h"
#include "worker_session.h"
#include "cuda/siglip2_megakernel.h"

// stb_image is already linked into siglip2_preproc as the implementation;
// here we only need extern declarations to call stbi_load_from_memory.
#include "stb_image.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <functional>
#include <future>
#include <thread>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

// ---- config -----------------------------------------------------------------

struct ServerConfig {
    std::string model_path;
    std::string tokenizer_path;
    std::string host = "0.0.0.0";
    int         port = 8890;
    int         default_max_num_patches = 729;
    int         idle_unload_seconds = 0;     // 0 = never
    bool        lazy_load = false;
    // Narrow-deployment switches: each saves ~1 GiB Q8 by skipping the
    // unused tower at load. Endpoints that require the missing tower
    // return 503.
    bool        vision_only = false;
    bool        text_only   = false;
    // worker-isolation: when set (via "--worker <fd>"), the process role
    // flips from HTTP server to subprocess worker dispatch loop. The fd
    // is the already-open Unix socketpair end inherited from the parent.
    int         worker_fd   = -1;
};

ServerConfig parse_args(int argc, char ** argv) {
    ServerConfig c;
    auto env = [](const char * key, const char * def) {
        const char * v = std::getenv(key);
        return v ? std::string(v) : std::string(def);
    };
    auto bool_env = [&](const char * key) {
        std::string v = env(key, "");
        return !v.empty() && v != "0" && v != "false";
    };
    c.model_path     = env("MODEL_PATH", "");
    c.tokenizer_path = env("TOKENIZER_PATH", "");
    c.host           = env("HOST", "0.0.0.0");
    c.port           = std::atoi(env("PORT", "8890").c_str());
    c.default_max_num_patches = std::atoi(env("DEFAULT_MAX_NUM_PATCHES", "729").c_str());
    c.idle_unload_seconds     = std::atoi(env("IDLE_UNLOAD_SECONDS", "0").c_str());
    c.lazy_load               = bool_env("LAZY_LOAD");
    c.vision_only             = bool_env("VISION_ONLY");
    c.text_only               = bool_env("TEXT_ONLY");

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](const char * flag, std::string & dst) {
            if (a == flag && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        if      (take("--model", c.model_path)) {}
        else if (take("--tokenizer", c.tokenizer_path)) {}
        else if (take("--host", c.host)) {}
        else if (a == "--port" && i + 1 < argc) c.port = std::atoi(argv[++i]);
        else if (a == "--default-max-num-patches" && i + 1 < argc) c.default_max_num_patches = std::atoi(argv[++i]);
        else if (a == "--idle-unload-seconds"     && i + 1 < argc) c.idle_unload_seconds     = std::atoi(argv[++i]);
        else if (a == "--lazy-load")   c.lazy_load   = true;
        else if (a == "--vision-only") c.vision_only = true;
        else if (a == "--text-only")   c.text_only   = true;
        else if (a == "--worker" && i + 1 < argc) {
            // Internal flag — set by spawn_worker() in the parent process.
            c.worker_fd = std::atoi(argv[++i]);
        }
        else if (a == "--help" || a == "-h") {
            fprintf(stderr,
                "siglip2-server — kobbler-vision-compatible HTTP service\n\n"
                "Usage: %s --model <gguf> --tokenizer <spm.model> [--port 8890] [--lazy-load]\n"
                "         [--vision-only | --text-only]\n"
                "Env: MODEL_PATH TOKENIZER_PATH HOST PORT DEFAULT_MAX_NUM_PATCHES\n"
                "     IDLE_UNLOAD_SECONDS LAZY_LOAD VISION_ONLY TEXT_ONLY\n"
                "     SIGLIP2_WORKER_ISOLATION=0  (opt out of subprocess worker; default ON)\n",
                argv[0]);
            std::exit(0);
        }
    }
    if (c.vision_only && c.text_only) {
        fprintf(stderr, "--vision-only and --text-only are mutually exclusive\n");
        std::exit(2);
    }
    return c;
}

// ---- L2 normalize -----------------------------------------------------------

void l2_normalize(std::vector<float> & v) {
    double s = 0.0;
    for (float x : v) s += (double)x * x;
    const float inv = (float)(1.0 / (std::sqrt(s) + 1e-12));
    for (auto & x : v) x *= inv;
}

// ---- ServerState ------------------------------------------------------------

// Persistent single-thread worker. Used by /v1/classify to fan out the text
// encode while vision runs on the request thread; using std::async per
// request was paying ~12 ms of CUDA driver per-thread init on every call,
// which ate most of the parallel-streams win. Holding the worker thread
// open keeps that init cost off the hot path.
struct PersistentWorker {
    std::mutex                m;
    std::condition_variable   cv_req;
    std::condition_variable   cv_done;
    std::function<void()>     task;
    bool                      pending  = false;   // true between submit and f()-return
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
    // Returns only AFTER the submitted lambda has returned. Earlier version
    // waited on `task == nullptr`, but the worker sets task to null BEFORE
    // calling the lambda — so wait_done could return mid-encode, the handler
    // would read uninitialized text_ok/text_err, and 500-with-empty-error
    // popped up under concurrent /v1/classify.
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

struct ServerState {
    ServerConfig                cfg;
    // Each encoder is single-threaded internally (gallocr buffers + tensor
    // pointers aren't reentrant), but vision and text run on PRIVATE
    // ggml_backend_t = private CUDA streams, so concurrent calls into the two
    // separate mutexes overlap on the GPU. /v1/classify uses this to fan out
    // text encode onto the persistent worker thread while vision runs on the
    // request thread; load_model holds both locks for the lifecycle transition.
    std::mutex                  vision_mutex;
    std::mutex                  text_mutex;
    // PersistentWorker has a single-task slot; concurrent classify handlers
    // would clobber each other's submitted lambdas. Each handler holds this
    // across the submit + wait_done block so worker access is serialized,
    // while still letting vision (this thread) and text (worker thread) run
    // concurrently within ONE classify call.
    std::mutex                  text_worker_mutex;
    PersistentWorker            text_worker;
    std::unique_ptr<siglip2::VisionEncoder> vision;
    std::unique_ptr<siglip2::TextEncoder>   text;
    std::unique_ptr<siglip2::Tokenizer>     tokenizer;
    siglip2::ScoreParams        score_params;
    std::atomic<bool>           loaded{false};
    std::atomic<long long>      last_request_ms{0};

    bool load_model(std::string & err) {
        std::scoped_lock lock(vision_mutex, text_mutex);
        if (loaded.load()) return true;
        if (cfg.model_path.empty()) {
            err = "model path required";
            return false;
        }
        const bool need_text = !cfg.vision_only;
        if (need_text && cfg.tokenizer_path.empty()) {
            err = "tokenizer path required (or pass --vision-only)";
            return false;
        }
        // Both towers loaded → give each its own backend so /v1/classify
        // overlaps them on separate CUDA streams. Single-tower deployments
        // ride the shared singleton (no peer to overlap with).
        const bool dual_tower = !cfg.vision_only && !cfg.text_only;
        if (!cfg.text_only) {
            vision = std::make_unique<siglip2::VisionEncoder>();
            if (!vision->load(cfg.model_path, /*private_backend=*/dual_tower)) {
                err = "vision load: " + vision->last_error();
                return false;
            }
        }
        if (need_text) {
            text      = std::make_unique<siglip2::TextEncoder>();
            tokenizer = std::make_unique<siglip2::Tokenizer>();
            if (!text->load(cfg.model_path, /*private_backend=*/dual_tower)) {
                err = "text load: " + text->last_error();
                return false;
            }
            if (!tokenizer->load(cfg.tokenizer_path)) {
                err = "tokenizer load: " + tokenizer->last_error();
                return false;
            }
        }
        // Score params (logit_scale/bias) are only needed when both towers are
        // present (classify endpoints); they're always cheap to read either way.
        if (dual_tower) {
            if (!siglip2::read_score_params(cfg.model_path, score_params, err)) {
                return false;
            }
        }
        loaded.store(true);
        if (dual_tower) text_worker.start();
        const char * mode = cfg.vision_only ? " (vision-only)" :
                            cfg.text_only   ? " (text-only)"   : "";
        fprintf(stderr, "[siglip2-server] model loaded: %s%s%s\n",
                cfg.model_path.c_str(), mode,
                dual_tower ? " [private-streams]" : "");
        siglip2_megakernel::log_device_stream_priority_range();
        if (dual_tower) {
            int prio_v = -1;
            if (const char * v = std::getenv("SIGLIP2_VISION_STREAM_PRIORITY")) prio_v = std::atoi(v);
            fprintf(stderr,
                "[siglip2-server] requested stream priorities: vision=%d (%s) text=0 (DEFAULT)\n",
                prio_v,
                prio_v < 0 ? "HIGH" : prio_v > 0 ? "LOW" : "DEFAULT");
        }
        if (siglip2_megakernel::profile_enabled()) {
            fprintf(stderr, "[siglip2-server] SIGLIP2_PROFILE_OPS=1 — per-anchor breakdown will be logged after each encode\n");
        }
        return true;
    }

    void unload() {
        const bool timing = std::getenv("SIGLIP2_TIME_HANDLER") != nullptr;
        auto tnow = []{ return std::chrono::steady_clock::now(); };
        auto tms  = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        auto t0 = tnow();
        std::scoped_lock lock(vision_mutex, text_mutex);
        auto t_lock = tnow();
        if (!loaded.load()) return;
        vision.reset();
        auto t_v = tnow();
        text.reset();
        auto t_t = tnow();
        tokenizer.reset();
        auto t_tk = tnow();
        loaded.store(false);
        if (timing) {
            fprintf(stderr, "[siglip2-server] unload timings (ms): lock=%.1f vision=%.1f text=%.1f tokenizer=%.1f total=%.1f\n",
                    tms(t0,t_lock), tms(t_lock,t_v), tms(t_v,t_t), tms(t_t,t_tk), tms(t0,t_tk));
        }
        fprintf(stderr, "[siglip2-server] model unloaded\n");
    }

    long long now_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void touch() { last_request_ms.store(now_ms()); }
};

// ---- helpers ----------------------------------------------------------------

// Decode an image (jpeg/png/...) from a memory buffer. Resulting RGB is uint8 row-major HWC.
bool decode_image(const std::string & bytes, std::vector<uint8_t> & rgb, int & h, int & w, std::string & err) {
    int channels = 0;
    uint8_t * px = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc *>(bytes.data()), (int)bytes.size(),
        &w, &h, &channels, /*req_comp=*/3);
    if (!px) {
        err = std::string("stbi_load_from_memory failed: ") + (stbi_failure_reason() ? stbi_failure_reason() : "?");
        return false;
    }
    rgb.assign(px, px + (size_t)h * w * 3);
    stbi_image_free(px);
    return true;
}

bool encode_image(
    ServerState &       st,
    const std::string & bytes,
    int                 max_num_patches,
    siglip2::Pooling    pooling,
    std::vector<float> & out_emb,
    std::string &       err) {
    static const bool timing = std::getenv("SIGLIP2_TIME_HANDLER") != nullptr;
    auto tnow = []{ return std::chrono::steady_clock::now(); };
    auto tms  = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = tnow();

    std::vector<uint8_t> rgb;
    int h = 0, w = 0;
    if (!decode_image(bytes, rgb, h, w, err)) return false;
    auto t_decode = tnow();

    siglip2::PreprocResult pp;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std_v[3] = {0.5f, 0.5f, 0.5f};
    if (!siglip2::preprocess_image_rgb(rgb.data(), h, w, max_num_patches,
                                       st.vision->config().patch_size,
                                       1.0f / 255.0f, mean, std_v, pp, err)) {
        return false;
    }
    auto t_preproc = tnow();

    if (!st.vision->encode(pp.pixel_values.data(), pp.n_patches_h, pp.n_patches_w, pooling, out_emb)) {
        err = "vision encode: " + st.vision->last_error();
        return false;
    }
    auto t_encode = tnow();

    if (timing) {
        std::fprintf(stderr,
            "[encode_image %dx%d→%dx%d] decode=%.2f preproc=%.2f encode=%.2f total=%.2f ms\n",
            w, h, pp.n_patches_w, pp.n_patches_h,
            tms(t0, t_decode), tms(t_decode, t_preproc),
            tms(t_preproc, t_encode), tms(t0, t_encode));
    }
    return true;
}

// Batched text encode.
//
// Tokenization matches HuggingFace Siglip2Processor: pad each prompt to
// max_position_embeddings (64) with pad_token_id, no attention_mask passed
// to the model. The live kobbler-vision service does `model.text_model(**inputs)`
// with input_ids only; pad-token-0 embeddings flow through attention as
// real tokens. Passing a mask would diverge by ~0.24 cosine on short prompts.
//
// All prompts run in ONE text-encoder graph compute (n_pos, n_batch) instead
// of N sequential graphs. The win is launch-overhead amortization plus better
// matmul utilization at higher M.
bool encode_text_batch(
    ServerState &                       st,
    const std::vector<std::string> &    prompts,
    std::vector<std::vector<float>> &   out_embs,
    std::string &                       err) {
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

// Collect form values for `name` from both multipart fields (req.files) and
// form-urlencoded body / URL params (req.params). FastAPI's list[str] = Form(...)
// can arrive either way depending on Content-Type the client sends.
std::vector<std::string> collect_field(const httplib::Request & req, const std::string & name) {
    std::vector<std::string> out;
    auto fr = req.files.equal_range(name);
    for (auto it = fr.first; it != fr.second; ++it) {
        out.push_back(it->second.content);
    }
    auto pr = req.params.equal_range(name);
    for (auto it = pr.first; it != pr.second; ++it) {
        out.push_back(it->second);
    }
    return out;
}

siglip2::Pooling parse_pooling(const std::string & s, std::string & err) {
    if (s.empty() || s == "pooler" || s == "probe") return siglip2::Pooling::PROBE;
    if (s == "mean") return siglip2::Pooling::MEAN;
    err = "pooling must be one of: pooler, probe, mean (got '" + s + "')";
    return siglip2::Pooling::PROBE;
}

void send_json(httplib::Response & res, int status, const json & body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

} // namespace

int main(int argc, char ** argv) {
    ServerConfig cfg = parse_args(argc, argv);

    // worker-isolation: when --worker <fd> was passed, this process is the
    // GPU child. Hand off to the dispatch loop and exit. The parent's
    // LOAD_REQ carries model_path/tokenizer_path/etc., so we don't validate
    // them here.
    if (cfg.worker_fd >= 0) {
        return siglip2::run_worker_loop(cfg.worker_fd);
    }

    if (cfg.model_path.empty()) {
        fprintf(stderr, "--model (or MODEL_PATH env) is required.\n");
        return 2;
    }
    if (!cfg.vision_only && cfg.tokenizer_path.empty()) {
        fprintf(stderr, "--tokenizer (or TOKENIZER_PATH env) is required unless --vision-only.\n");
        return 2;
    }

    // worker-isolation: GPU work runs in a forked child. The parent never
    // touches CUDA, so /v1/admin/unload SIGKILLs the worker and reclaims
    // ALL VRAM (CUDA primary context, kernel cache, ggml-cuda buffers —
    // not just the in-process heap). Default ON; SIGLIP2_WORKER_ISOLATION=0
    // forces single-process mode (e.g. for parity benches against the
    // worker-iso path or when a tool wants to embed siglip2 directly).
    bool worker_iso = true;
    if (const char * env = std::getenv("SIGLIP2_WORKER_ISOLATION")) {
        worker_iso = env[0] && env[0] != '0';
    }
    std::unique_ptr<siglip2::WorkerSession> worker_session;
    siglip2::WorkerLoadConfig worker_cfg;
    auto fill_worker_cfg = [&]() {
        worker_cfg.model_path              = cfg.model_path;
        worker_cfg.tokenizer_path          = cfg.tokenizer_path;
        worker_cfg.default_max_num_patches = cfg.default_max_num_patches;
        worker_cfg.vision_only             = cfg.vision_only;
        worker_cfg.text_only               = cfg.text_only;
        worker_cfg.lazy_load               = false;
    };
    fill_worker_cfg();

    if (worker_iso) {
        // Forward CLI args except --worker (set by spawn_worker itself) so
        // the child sees the same model/tokenizer/--*-only options.
        std::vector<std::string> child_argv;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--worker" && i + 1 < argc) { i++; continue; }
            child_argv.push_back(a);
        }
        worker_session = std::make_unique<siglip2::WorkerSession>(argv[0], child_argv);
        // Default GPU (UUID) for un-targeted (direct / unbounded) requests.
        // Per-request `gpu` field overrides. Empty = inherit container
        // CUDA_VISIBLE_DEVICES. Standard env name across all kob services.
        if (const char * g = std::getenv("WORKER_DEFAULT_GPU")) {
            worker_session->set_default_gpu(g);
            fprintf(stderr, "[siglip2-server] default GPU: %s\n", g);
        }
        fprintf(stderr, "[siglip2-server] worker-isolation: ON (subprocess owns GPU; "
                        "/v1/admin/unload SIGKILLs the worker → 0 VRAM)\n");
    } else {
        // In-process mode (SIGLIP2_WORKER_ISOLATION=0): install fused CUDA
        // kernels. Must run before any encoder load so the hooks are live
        // for the first graph compute. No-op when GGML_CUDA is OFF or
        // SIGLIP2_DISABLE_MEGAKERNEL=1.
        fprintf(stderr, "[siglip2-server] worker-isolation: OFF (in-process; "
                        "SIGLIP2_WORKER_ISOLATION=0 set)\n");
        siglip2_megakernel::install();
    }

    ServerState st;
    st.cfg = cfg;
    if (!worker_session && !cfg.lazy_load) {
        std::string err;
        if (!st.load_model(err)) {
            fprintf(stderr, "load failed: %s\n", err.c_str());
            return 1;
        }
        st.touch();
    }
    if (worker_session && !cfg.lazy_load) {
        if (!worker_session->ensure_loaded(worker_cfg)) {
            fprintf(stderr, "fatal: worker model load failed: %s\n",
                    worker_session->last_error().c_str());
            return 1;
        }
        fprintf(stderr, "[siglip2-server] worker pid=%d loaded model OK\n",
                (int) worker_session->pid());
        st.touch();
    }

    httplib::Server srv;
    srv.set_keep_alive_max_count(20);
    srv.set_read_timeout(60);
    srv.set_write_timeout(60);

    // cpp-httplib v0.20 only short-circuits DELETE without Content-Length;
    // POST/PUT/PATCH with no Content-Length and no Transfer-Encoding deadlock
    // in read_content_without_length until the keep-alive timeout fires (5s)
    // and the connection drops, returning 400. /unload, /v1/admin/load, and
    // other admin POSTs are body-less by design — internal callers (koblem,
    // curl without -d) hit this constantly. Inject Content-Length: 0 so the
    // body read returns immediately with len==0.
    srv.set_pre_routing_handler([](const httplib::Request & req, httplib::Response &) {
        if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") &&
            !req.has_header("Content-Length") &&
            req.get_header_value("Transfer-Encoding") != "chunked") {
            const_cast<httplib::Request &>(req).set_header("Content-Length", "0");
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // -- /health
    srv.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        const bool loaded = worker_session ? worker_session->is_alive()
                                           : st.loaded.load();
        json body = {
            {"status", "ok"},
            {"model", cfg.model_path},
            {"model_loaded", loaded},
        };
        if (worker_session) body["worker_pid"] = worker_session->pid();
        send_json(res, 200, body);
    });

    // -- /v1/gpu/status : GPU residency announce for the koblem gate.
    srv.Get("/v1/gpu/status", [&](const httplib::Request &, httplib::Response & res) {
        const bool loaded = worker_session ? worker_session->is_alive()
                                           : st.loaded.load();
        json body = {{"loaded", loaded}};
        if (loaded && worker_session && !worker_session->worker_gpu().empty())
            body["gpu"] = worker_session->worker_gpu();
        else
            body["gpu"] = nullptr;
        send_json(res, 200, body);
    });

    // ensure-model helper. In worker mode: spawn + LOAD if not running.
    // In in-process mode: load encoder + tokenizer + score params.
    auto ensure = [&](httplib::Response & res) -> bool {
        st.touch();
        if (worker_session) {
            if (!worker_session->ensure_loaded(worker_cfg)) {
                send_json(res, 503, json{{"error", worker_session->last_error()}});
                return false;
            }
            return true;
        }
        if (!st.loaded.load()) {
            std::string err;
            if (!st.load_model(err)) {
                send_json(res, 503, json{{"error", err}});
                return false;
            }
        }
        return true;
    };

    // -- /unload (legacy) and /v1/admin/unload — both SIGKILL the worker
    // when worker_session is active so VRAM is fully reclaimed (kernel
    // cache + cuBLAS workspace + ggml-cuda buffers). In-process mode just
    // resets the encoder unique_ptrs.
    auto do_unload = [&](httplib::Response & res, const char * source) {
        bool was_loaded = false;
        if (worker_session) {
            was_loaded = worker_session->is_alive();
            worker_session->shutdown();
        } else {
            was_loaded = st.loaded.load();
            if (was_loaded) st.unload();
        }
        const bool now_loaded = worker_session ? worker_session->is_alive()
                                               : st.loaded.load();
        json out = {{"unloaded", was_loaded}, {"model_loaded", now_loaded},
                    {"source", source}};
        if (worker_session) out["mode"] = "worker-isolation";
        send_json(res, 200, out);
    };
    srv.Post("/unload", [&](const httplib::Request &, httplib::Response & res) {
        do_unload(res, "/unload");
    });
    srv.Post("/v1/admin/unload", [&](const httplib::Request &, httplib::Response & res) {
        do_unload(res, "/v1/admin/unload");
    });

    // -- /v1/admin/load — explicit reload. In worker mode this is a
    // synchronous spawn + LOAD_REQ; in in-process mode it loads the
    // encoder + tokenizer.
    srv.Post("/v1/admin/load", [&](const httplib::Request &, httplib::Response & res) {
        bool was_loaded = false;
        bool ok = false;
        std::string err_msg;
        if (worker_session) {
            was_loaded = worker_session->is_alive();
            ok = worker_session->ensure_loaded(worker_cfg);
            if (!ok) err_msg = worker_session->last_error();
        } else {
            was_loaded = st.loaded.load();
            ok = was_loaded || st.load_model(err_msg);
        }
        if (!ok) {
            send_json(res, 500, json{{"error", err_msg}});
            return;
        }
        const bool now_loaded = worker_session ? worker_session->is_alive()
                                               : st.loaded.load();
        json out = {{"loaded", !was_loaded}, {"model_loaded", now_loaded}};
        if (worker_session) out["mode"] = "worker-isolation";
        send_json(res, 200, out);
    });

    auto require_vision = [&](httplib::Response & res) -> bool {
        // In worker mode the encoders live in the child; gate on cfg flags.
        const bool have_vision = worker_session ? !cfg.text_only : (st.vision != nullptr);
        if (!have_vision) {
            send_json(res, 503, json{{"error", "vision tower not loaded (server started with --text-only)"}});
            return false;
        }
        return true;
    };
    auto require_text = [&](httplib::Response & res) -> bool {
        const bool have_text = worker_session ? !cfg.vision_only
                                              : (st.text != nullptr && st.tokenizer != nullptr);
        if (!have_text) {
            send_json(res, 503, json{{"error", "text tower not loaded (server started with --vision-only)"}});
            return false;
        }
        return true;
    };
    auto require_score = [&](httplib::Response & res) -> bool {
        const bool have_both = worker_session ? (!cfg.text_only && !cfg.vision_only)
                                              : (st.vision != nullptr && st.text != nullptr && st.tokenizer != nullptr);
        if (!have_both) {
            send_json(res, 503, json{{"error", "classify requires both towers loaded"}});
            return false;
        }
        return true;
    };

    // -- /v1/embeddings
    srv.Post("/v1/embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ensure(res)) return;
        if (!require_vision(res)) return;
        auto images = req.get_file_values("images");
        if (images.empty()) {
            send_json(res, 400, json{{"error", "No images provided"}});
            return;
        }
        std::string err;
        const std::string pooling_str = req.get_param_value("pooling");
        siglip2::Pooling pooling = parse_pooling(pooling_str, err);
        if (!err.empty()) { send_json(res, 400, json{{"error", err}}); return; }
        int max_num_patches = cfg.default_max_num_patches;
        if (req.has_param("max_num_patches")) {
            const std::string v = req.get_param_value("max_num_patches");
            if (!v.empty()) max_num_patches = std::atoi(v.c_str());
        }
        const bool return_last_hidden = req.get_param_value("return_last_hidden") == "true";
        if (return_last_hidden) {
            // Not implemented in M4 — would require a separate encode-with-hidden path.
            send_json(res, 501, json{{"error", "return_last_hidden not implemented"}});
            return;
        }

        // `gpu` may arrive as a multipart field (alongside images) or a query
        // param — collect_field() reads both. Empty → default/inherited card.
        auto gpu_f = collect_field(req, "gpu");
        const std::string gpu = gpu_f.empty() ? std::string() : gpu_f.front();

        std::vector<std::vector<float>> embs;
        if (worker_session) {
            std::vector<std::string> img_bytes;
            img_bytes.reserve(images.size());
            for (const auto & img : images) img_bytes.push_back(img.content);
            if (!worker_session->encode_images(img_bytes, pooling_str.empty() ? "probe" : pooling_str,
                                               max_num_patches, /*return_last_hidden=*/false, embs, gpu)) {
                send_json(res, 500, json{{"error", worker_session->last_error()}});
                return;
            }
        } else {
            embs.reserve(images.size());
            std::lock_guard<std::mutex> lock(st.vision_mutex);
            for (const auto & img : images) {
                std::vector<float> emb;
                if (!encode_image(st, img.content, max_num_patches, pooling, emb, err)) {
                    send_json(res, 500, json{{"error", err}});
                    return;
                }
                l2_normalize(emb);
                embs.emplace_back(std::move(emb));
            }
        }
        send_json(res, 200, json{{"embeddings", embs}});
    });

    // -- /v1/text_embeddings
    srv.Post("/v1/text_embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        if (!ensure(res)) return;
        if (!require_text(res)) return;
        // Accept multiple `prompts` form fields per kobbler-vision (FastAPI list[str] = Form(...)).
        std::vector<std::string> prompts = collect_field(req, "prompts");
        if (prompts.empty()) { send_json(res, 400, json{{"error", "No prompts provided"}}); return; }

        std::vector<std::vector<float>> embs;
        if (worker_session) {
            if (!worker_session->encode_texts(prompts, embs)) {
                send_json(res, 500, json{{"error", worker_session->last_error()}});
                return;
            }
            // Worker already L2-normalizes.
        } else {
            std::string err;
            {
                std::lock_guard<std::mutex> lock(st.text_mutex);
                if (!encode_text_batch(st, prompts, embs, err)) {
                    send_json(res, 500, json{{"error", err}});
                    return;
                }
            }
            for (auto & e : embs) l2_normalize(e);
        }
        send_json(res, 200, json{{"embeddings", embs}});
    });

    // -- /v1/classify
    srv.Post("/v1/classify", [&](const httplib::Request & req, httplib::Response & res) {
        const bool timing = std::getenv("SIGLIP2_TIME_HANDLER") != nullptr;
        auto tnow = []{ return std::chrono::steady_clock::now(); };
        auto tms  = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        auto t0 = tnow();
        if (!ensure(res)) return;
        if (!require_score(res)) return;
        auto images = req.get_file_values("images");
        std::vector<std::string> prompts = collect_field(req, "prompts");
        if (images.empty())  { send_json(res, 400, json{{"error", "No images provided"}}); return; }
        if (prompts.empty()) { send_json(res, 400, json{{"error", "No prompts provided"}}); return; }

        const bool return_logits = req.get_param_value("return_logits") == "true";
        int max_num_patches = cfg.default_max_num_patches;
        if (req.has_param("max_num_patches")) {
            const std::string v = req.get_param_value("max_num_patches");
            if (!v.empty()) max_num_patches = std::atoi(v.c_str());
        }

        if (worker_session) {
            std::vector<std::string> img_bytes;
            img_bytes.reserve(images.size());
            for (const auto & img : images) img_bytes.push_back(img.content);
            std::vector<std::vector<float>> scores_out, logits_out;
            if (!worker_session->classify(img_bytes, prompts, max_num_patches,
                                          return_logits, scores_out, logits_out)) {
                send_json(res, 500, json{{"error", worker_session->last_error()}});
                return;
            }
            json body = {{"scores", scores_out}};
            if (return_logits) body["logits"] = logits_out;
            send_json(res, 200, body);
            return;
        }

        const int H = (int)st.vision->config().hidden_size;
        const int n_img = (int)images.size();
        const int n_txt = (int)prompts.size();
        std::vector<float> img_buf((size_t)n_img * H);
        std::vector<float> txt_buf((size_t)n_txt * H);
        auto t_parse = tnow();

        // Run vision and text encodes on parallel CUDA streams. With private
        // ggml_backend_t per encoder (set up in load_model) the two graph
        // computes overlap on the GPU; the text encode is dispatched to the
        // persistent worker thread (avoiding per-call CUDA driver init that
        // std::async would pay), while vision runs on the request thread.
        // Each branch owns its own encoder mutex, so concurrent requests
        // still serialize per-encoder.
        // Hold text_worker_mutex across submit + wait_done so concurrent
        // classify handlers don't clobber each other's submitted lambdas
        // (the worker has a single-task slot). Worker still runs the lambda
        // on its own thread → text overlaps vision on private CUDA streams.
        std::lock_guard<std::mutex> worker_lock(st.text_worker_mutex);

        std::string text_err;
        std::vector<std::vector<float>> txt_embs;
        bool text_ok = false;
        std::chrono::steady_clock::time_point t_text_start, t_text_done;
        st.text_worker.submit([&]() {
            t_text_start = tnow();
            std::lock_guard<std::mutex> lock(st.text_mutex);
            text_ok = encode_text_batch(st, prompts, txt_embs, text_err);
            t_text_done = tnow();
        });
        auto t_submit = tnow();

        std::string err;
        bool vision_ok = true;
        {
            std::lock_guard<std::mutex> lock(st.vision_mutex);
            for (int i = 0; i < n_img; ++i) {
                std::vector<float> emb;
                if (!encode_image(st, images[i].content, max_num_patches, siglip2::Pooling::PROBE, emb, err)) {
                    vision_ok = false;
                    break;
                }
                std::memcpy(img_buf.data() + (size_t)i * H, emb.data(), sizeof(float) * H);
            }
        }
        auto t_vision_done = tnow();
        st.text_worker.wait_done();
        auto t_join = tnow();
        if (!vision_ok) { send_json(res, 500, json{{"error", err}}); return; }
        if (!text_ok)   { send_json(res, 500, json{{"error", text_err}}); return; }
        for (int j = 0; j < n_txt; ++j) {
            std::memcpy(txt_buf.data() + (size_t)j * H, txt_embs[j].data(), sizeof(float) * H);
        }
        std::vector<float> logits((size_t)n_img * n_txt);
        std::vector<float> probs((size_t)n_img * n_txt);
        siglip2::score_image_text(
            img_buf.data(), n_img, txt_buf.data(), n_txt, H,
            st.score_params, logits.data(), probs.data());

        // scores: (n_img, n_txt)
        std::vector<std::vector<float>> scores_out(n_img, std::vector<float>(n_txt));
        std::vector<std::vector<float>> logits_out(n_img, std::vector<float>(n_txt));
        for (int i = 0; i < n_img; ++i)
            for (int j = 0; j < n_txt; ++j) {
                scores_out[i][j] = probs[(size_t)i * n_txt + j];
                logits_out[i][j] = logits[(size_t)i * n_txt + j];
            }
        json body = {{"scores", scores_out}};
        if (return_logits) body["logits"] = logits_out;
        auto t_score = tnow();
        send_json(res, 200, body);
        auto t_send = tnow();
        if (timing) {
            fprintf(stderr,
                "[/v1/classify] parse=%.2f submit=%.2f vision=%.2f join_after_vision=%.2f "
                "text(%.2f→%.2f=%.2f) score+pack=%.2f json=%.2f total=%.2f\n",
                tms(t0,t_parse), tms(t_parse,t_submit), tms(t_submit,t_vision_done),
                tms(t_vision_done,t_join),
                tms(t_submit,t_text_start), tms(t_text_start,t_text_done),
                tms(t_text_start,t_text_done),
                tms(t_join,t_score), tms(t_score,t_send), tms(t0,t_send));
        }
    });

    // -- /v1/classify_from_embeddings (JSON body)
    srv.Post("/v1/classify_from_embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        const bool timing = std::getenv("SIGLIP2_TIME_HANDLER") != nullptr;
        auto tnow = []{ return std::chrono::steady_clock::now(); };
        auto tms  = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        auto t0 = tnow();

        if (!ensure(res)) return;
        // image embeddings are supplied by the client, but we still need text + score.
        if (!require_text(res)) return;
        const bool have_vision = worker_session ? !cfg.text_only : (st.vision != nullptr);
        if (!have_vision) {
            send_json(res, 503, json{{"error", "classify requires the model loaded with both towers"}});
            return;
        }
        json body;
        try {
            body = json::parse(req.body);
        } catch (std::exception & e) {
            send_json(res, 400, json{{"error", std::string("invalid JSON: ") + e.what()}});
            return;
        }
        if (!body.contains("image_embeddings") || !body.contains("prompts")) {
            send_json(res, 400, json{{"error", "JSON must contain image_embeddings and prompts"}});
            return;
        }
        auto t_parse = tnow();
        const bool return_logits = body.value("return_logits", false);

        const int H = worker_session ? worker_session->hidden_size()
                                     : (int)st.vision->config().hidden_size;
        const auto & imgs_json = body["image_embeddings"];
        const auto & prompts_json = body["prompts"];
        const int n_img = (int)imgs_json.size();
        const int n_txt = (int)prompts_json.size();

        if (worker_session) {
            std::vector<std::vector<float>> img_embs(n_img, std::vector<float>(H));
            for (int i = 0; i < n_img; ++i) {
                const auto & row = imgs_json[i];
                if ((int)row.size() != H) {
                    send_json(res, 400, json{{"error", "image_embedding row has wrong dim"}});
                    return;
                }
                for (int c = 0; c < H; ++c) img_embs[i][c] = row[c].get<float>();
            }
            std::vector<std::string> prompts(n_txt);
            for (int j = 0; j < n_txt; ++j) prompts[j] = prompts_json[j].get<std::string>();
            std::vector<std::vector<float>> scores_out, logits_out;
            if (!worker_session->classify_from_embeddings(img_embs, prompts, return_logits,
                                                          scores_out, logits_out)) {
                send_json(res, 500, json{{"error", worker_session->last_error()}});
                return;
            }
            json out = {{"scores", scores_out}};
            if (return_logits) out["logits"] = logits_out;
            send_json(res, 200, out);
            auto t_done = tnow();
            if (timing) {
                std::fprintf(stderr,
                    "[cfe] parse=%.2f total(worker)=%.2f ms\n",
                    tms(t0, t_parse), tms(t0, t_done));
            }
            return;
        }

        std::vector<float> img_buf((size_t)n_img * H);
        for (int i = 0; i < n_img; ++i) {
            const auto & row = imgs_json[i];
            if ((int)row.size() != H) {
                send_json(res, 400, json{{"error", "image_embedding row has wrong dim"}});
                return;
            }
            for (int c = 0; c < H; ++c) img_buf[(size_t)i * H + c] = row[c].get<float>();
        }
        auto t_imgs = tnow();

        std::vector<float> txt_buf((size_t)n_txt * H);
        std::string err;
        {
            std::lock_guard<std::mutex> lock(st.text_mutex);
            std::vector<std::string> prompts(n_txt);
            for (int j = 0; j < n_txt; ++j) prompts[j] = prompts_json[j].get<std::string>();
            std::vector<std::vector<float>> txt_embs;
            if (!encode_text_batch(st, prompts, txt_embs, err)) {
                send_json(res, 500, json{{"error", err}}); return;
            }
            for (int j = 0; j < n_txt; ++j) {
                std::memcpy(txt_buf.data() + (size_t)j * H, txt_embs[j].data(), sizeof(float) * H);
            }
        }
        auto t_encode = tnow();

        std::vector<float> logits((size_t)n_img * n_txt);
        std::vector<float> probs((size_t)n_img * n_txt);
        siglip2::score_image_text(
            img_buf.data(), n_img, txt_buf.data(), n_txt, H,
            st.score_params, logits.data(), probs.data());
        auto t_score = tnow();

        std::vector<std::vector<float>> scores_out(n_img, std::vector<float>(n_txt));
        std::vector<std::vector<float>> logits_out(n_img, std::vector<float>(n_txt));
        for (int i = 0; i < n_img; ++i)
            for (int j = 0; j < n_txt; ++j) {
                scores_out[i][j] = probs[(size_t)i * n_txt + j];
                logits_out[i][j] = logits[(size_t)i * n_txt + j];
            }
        json out = {{"scores", scores_out}};
        if (return_logits) out["logits"] = logits_out;
        send_json(res, 200, out);
        auto t_done = tnow();

        if (timing) {
            std::fprintf(stderr,
                "[cfe] parse=%.2f imgs=%.2f encode=%.2f score=%.2f respond=%.2f total=%.2f ms\n",
                tms(t0, t_parse), tms(t_parse, t_imgs), tms(t_imgs, t_encode),
                tms(t_encode, t_score), tms(t_score, t_done), tms(t0, t_done));
        }
    });

    // Idle-unload sweeper. Polls every 10 s; if the model has been loaded
    // and no real request has hit ensure() / touch() within the threshold,
    // SIGKILL the worker (or unload in-process state). /health intentionally
    // does NOT touch, so docker healthchecks don't reset the timer.
    // Detached — process exit reaps the thread.
    if (cfg.idle_unload_seconds > 0) {
        std::thread([&st, &worker_session, secs = cfg.idle_unload_seconds]() {
            const auto      poll      = std::chrono::seconds(10);
            const long long thresh_ms = (long long) secs * 1000;
            for (;;) {
                std::this_thread::sleep_for(poll);
                const bool loaded = worker_session ? worker_session->is_alive()
                                                   : st.loaded.load();
                if (!loaded) continue;
                const long long last = st.last_request_ms.load();
                if (last == 0) continue;  // never touched (paranoia: eager-load touches above)
                const long long age = st.now_ms() - last;
                if (age < thresh_ms) continue;
                fprintf(stderr,
                    "[siglip2-server] idle-unload: %lld s since last request "
                    "(threshold %d s)\n", age / 1000, secs);
                if (worker_session) {
                    worker_session->shutdown();
                } else {
                    st.unload();
                }
            }
        }).detach();
        fprintf(stderr,
            "[siglip2-server] idle-unload sweeper: %d s threshold, 10 s poll\n",
            cfg.idle_unload_seconds);
    }

    fprintf(stderr, "[siglip2-server] listening on %s:%d (model=%s)\n",
        cfg.host.c_str(), cfg.port, cfg.model_path.c_str());
    if (!srv.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "listen failed\n");
        return 1;
    }
    return 0;
}
