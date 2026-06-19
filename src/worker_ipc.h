// worker_ipc.h — length-prefixed frame protocol over Unix-domain
// socketpair, used by the siglip2-server subprocess-worker model.
//
// Parent role: HTTP server only, no GPU. Owns the parent end of the
// socket. Spawns child via fork()+execv("--worker <fd>").
//
// Worker role: owns VisionEncoder + TextEncoder + Tokenizer + Score
// params + ggml-cuda context. Started via fork()+execv from parent so
// the address space is fresh (no copy-on-write inheritance of parent
// state into the CUDA process). Reads frames from the inherited socket
// fd, dispatches.
//
// Wire format:
//
//   [u32 frame_type][u32 payload_len][u32 req_id][u8 payload[payload_len]]
//
// Most payloads are JSON; frames that carry raw float embeddings or
// raw image bytes use a [u32 json_len][json][blob] layout via the
// pack_blob_payload helper so multi-MiB float arrays don't get base64-
// stuffed through JSON.

#ifndef SIGLIP2_WORKER_IPC_H
#define SIGLIP2_WORKER_IPC_H

#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

namespace siglip2 {

enum class WorkerFrame : uint32_t {
    HELLO                       = 0x01,  // W→P  {"pid": int, "role": str}
    LOAD_REQ                    = 0x10,  // P→W  see WorkerLoadConfig
    LOAD_RESP                   = 0x11,  // W→P  {"ok": bool, "error": str, "hidden_size": int, "max_position_embeddings": int}
    ENCODE_IMAGES_REQ           = 0x20,  // P→W  json hdr + N image blobs
    ENCODE_IMAGES_RESP          = 0x21,  // W→P  json hdr + raw f32 embeddings
    ENCODE_TEXT_REQ             = 0x22,  // P→W  json {"prompts": [str]}
    ENCODE_TEXT_RESP            = 0x23,  // W→P  json hdr + raw f32 embeddings
    CLASSIFY_REQ                = 0x24,  // P→W  json hdr + N image blobs
    CLASSIFY_RESP               = 0x25,  // W→P  json {"scores": [[f]], "logits": [[f]]}
    CLASSIFY_FROM_EMB_REQ       = 0x26,  // P→W  json hdr + raw f32 image embeddings
    CLASSIFY_FROM_EMB_RESP      = 0x27,  // W→P  json {"scores": [[f]], "logits": [[f]]}
    ERR                         = 0x2F,  // W→P  {"error": str}  — generic per-request failure
    PING                        = 0x40,  // either {"t_send_ns": u64}
    PONG                        = 0x41,  // either {"t_send_ns": u64, "t_recv_ns": u64}
    SHUTDOWN                    = 0xFF,  // P→W   ask worker to exit cleanly
};

struct FrameHeader {
    uint32_t type;     // WorkerFrame
    uint32_t len;      // payload bytes that follow
    uint32_t req_id;   // 0 = unsolicited / no correlation
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader must stay 12 bytes");

inline constexpr size_t HEADER_BYTES      = sizeof(FrameHeader);
inline constexpr size_t MAX_FRAME_PAYLOAD = 256u * 1024u * 1024u; // 256 MiB safety cap

enum class IpcError {
    OK = 0,
    EofClean,
    EofMidFrame,
    SocketError,
    ProtocolError,
    PayloadTooBig,
};

const char * ipc_error_str(IpcError e);

IpcError read_exact(int fd, void * buf, size_t len);
IpcError write_exact(int fd, const void * buf, size_t len);

IpcError send_frame(int fd,
                    WorkerFrame type,
                    uint32_t req_id,
                    const void * payload,
                    size_t payload_len);

IpcError send_frame(int fd,
                    WorkerFrame type,
                    uint32_t req_id,
                    const std::string & json);

IpcError send_frame(int fd,
                    WorkerFrame type,
                    uint32_t req_id,
                    const std::vector<uint8_t> & payload);

IpcError recv_frame(int fd,
                    FrameHeader * out_hdr,
                    std::vector<uint8_t> * out_payload);

// [u32 json_len][json bytes][raw f32 bytes] — used by ENCODE_*_RESP.
std::vector<uint8_t> pack_blob_payload(const std::string & json_meta,
                                       const float * floats,
                                       size_t n_floats);

bool unpack_blob_payload(const std::vector<uint8_t> & payload,
                         std::string * out_meta,
                         std::vector<float> * out_floats);

// [u32 json_len][json bytes][per-blob: u32 len + bytes...] — used by
// ENCODE_IMAGES_REQ / CLASSIFY_REQ where the parent forwards N raw
// image byte strings to the worker.
std::vector<uint8_t> pack_multi_blob_payload(
    const std::string & json_meta,
    const std::vector<std::string> & blobs);

bool unpack_multi_blob_payload(
    const std::vector<uint8_t> & payload,
    std::string * out_meta,
    std::vector<std::string> * out_blobs);

// Spawn helper: socketpair() + fork() + execv(self_argv0, "--worker N").
// Returns the child pid on success and writes the parent-side fd to
// `*out_parent_fd`. Returns -1 on failure. `extra_argv` is appended after
// "--worker N" so the child sees the same load-time arguments.
//
// `cuda_visible_devices`, when non-empty, is set as CUDA_VISIBLE_DEVICES in
// the child *before* execv — pinning the worker to a specific physical GPU.
// The parent stays CUDA-free (worker-isolation), so changing it per-spawn is
// safe and lets a caller place / relocate the worker onto either card. Empty
// = inherit the parent/container environment (current default behaviour).
pid_t spawn_worker(const char * self_argv0,
                   const std::vector<std::string> & extra_argv,
                   int * out_parent_fd,
                   const std::string & cuda_visible_devices = "");

} // namespace siglip2

#endif // SIGLIP2_WORKER_IPC_H
