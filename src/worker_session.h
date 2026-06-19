// worker_session.h — parent-side handle on a subprocess worker, plus the
// child-side dispatch loop. The worker owns ALL GPU state (CUDA primary
// context, ggml backend buffers, encoder weights), and shutdown() SIGKILLs
// the child so the kernel reclaims every byte of VRAM the worker held.
//
// Parent path (HTTP server, no GPU): build args once, call
// ensure_loaded() lazily on the first request, route encode/classify
// calls through the four request methods. shutdown() is wired to the
// /v1/admin/unload endpoint; the next request after that respawns the
// worker via ensure_loaded().

#ifndef SIGLIP2_WORKER_SESSION_H
#define SIGLIP2_WORKER_SESSION_H

#include "worker_ipc.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

namespace siglip2 {

struct WorkerLoadConfig {
    std::string model_path;
    std::string tokenizer_path;
    int         default_max_num_patches = 729;
    bool        vision_only             = false;
    bool        text_only               = false;
    bool        lazy_load               = false;
};

class WorkerSession {
public:
    WorkerSession(const char * argv0,
                  std::vector<std::string> extra_argv = {});
    ~WorkerSession();

    // Spawn worker + send LOAD_REQ if not running, or if cfg changed.
    // Idempotent on identical cfg + alive worker.
    bool ensure_loaded(const WorkerLoadConfig & cfg);

    // SIGKILL + waitpid. Idempotent. Subsequent ensure_loaded() respawns.
    void shutdown();

    // Default GPU (CUDA_VISIBLE_DEVICES value: index or "GPU-..." UUID) for
    // workers spawned without a per-request override. Empty = inherit the
    // container env. Lets unbounded/direct requests land on the right card.
    void set_default_gpu(std::string gpu) { default_gpu_ = std::move(gpu); }

    bool         is_alive()      const { return pid_ > 0; }
    pid_t        pid()           const { return pid_; }
    const std::string & worker_gpu() const { return worker_gpu_; }
    int          hidden_size()   const { return hidden_size_; }
    int          max_position_embeddings() const { return max_pos_; }
    const std::string & last_error() const { return last_error_; }

    // Vision encode → per-image float embedding (L2-normalized). Pooling is
    // one of "probe" / "pooler" / "mean" — the worker resolves the string.
    // return_last_hidden currently unimplemented (worker returns ok=false).
    // `gpu` (when non-empty) overrides the target card for this request; if
    // the resident worker is on a different GPU it is relocated (kill+respawn).
    bool encode_images(
        const std::vector<std::string> & image_bytes,
        const std::string &              pooling,
        int                              max_num_patches,
        bool                             return_last_hidden,
        std::vector<std::vector<float>> & out_embs,
        const std::string &              gpu = std::string());

    // Text encode → per-prompt float embedding (L2-normalized).
    bool encode_texts(
        const std::vector<std::string> &  prompts,
        std::vector<std::vector<float>> & out_embs,
        const std::string &               gpu = std::string());

    // Full classify pipeline. out_scores / out_logits are (n_img × n_txt).
    bool classify(
        const std::vector<std::string> &  image_bytes,
        const std::vector<std::string> &  prompts,
        int                               max_num_patches,
        bool                              return_logits,
        std::vector<std::vector<float>> & out_scores,
        std::vector<std::vector<float>> & out_logits,
        const std::string &               gpu = std::string());

    // Score with externally-supplied image embeddings.
    bool classify_from_embeddings(
        const std::vector<std::vector<float>> & image_embeddings,
        const std::vector<std::string> &        prompts,
        bool                                    return_logits,
        std::vector<std::vector<float>> &       out_scores,
        std::vector<std::vector<float>> &       out_logits,
        const std::string &                     gpu = std::string());

private:
    bool send_load_req_locked(const WorkerLoadConfig & cfg);
    void kill_worker_locked();
    // Spawn + LOAD round-trip without taking io_mutex_ (caller holds it).
    // Used by ensure_loaded() and by the request methods to self-heal
    // when they observe the worker died during a racing /unload.
    // `want_gpu` empty → use default_gpu_. Respawns if the resident worker's
    // GPU differs from the resolved target (relocation).
    bool ensure_loaded_locked(const WorkerLoadConfig & cfg,
                              const std::string & want_gpu);
    // True if a request can proceed without respawning. Caller holds lock.
    bool worker_alive_locked() const { return pid_ > 0 && fd_ >= 0 && loaded_ok_; }

    std::string                argv0_;
    std::vector<std::string>   extra_argv_;
    WorkerLoadConfig           loaded_cfg_;
    bool                       loaded_ok_  = false;
    std::string                default_gpu_;   // CVD for un-targeted spawns
    std::string                worker_gpu_;    // GPU the live worker is pinned to

    pid_t                      pid_        = -1;
    int                        fd_         = -1;
    int                        hidden_size_ = 0;
    int                        max_pos_    = 0;
    mutable std::mutex         io_mutex_;
    std::string                last_error_;
    std::atomic<uint32_t>      next_req_id_{1};
};

// Worker-side dispatch loop. Called from main() when --worker <fd> is
// passed. Owns vision/text/tokenizer/score state, services LOAD_REQ +
// the four encode/classify frames, exits on EOF or SHUTDOWN.
int run_worker_loop(int fd);

} // namespace siglip2

#endif // SIGLIP2_WORKER_SESSION_H
