// worker_ipc.cpp — see worker_ipc.h.

#include "worker_ipc.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace siglip2 {

const char * ipc_error_str(IpcError e) {
    switch (e) {
        case IpcError::OK:             return "OK";
        case IpcError::EofClean:       return "peer closed cleanly";
        case IpcError::EofMidFrame:    return "peer closed mid-frame";
        case IpcError::SocketError:    return "socket error";
        case IpcError::ProtocolError:  return "protocol error";
        case IpcError::PayloadTooBig:  return "payload too big";
    }
    return "unknown";
}

IpcError read_exact(int fd, void * buf, size_t len) {
    char * p = static_cast<char *>(buf);
    size_t got = 0;
    while (got < len) {
        ssize_t r = ::read(fd, p + got, len - got);
        if (r > 0) { got += static_cast<size_t>(r); continue; }
        if (r == 0) {
            return got == 0 ? IpcError::EofClean : IpcError::EofMidFrame;
        }
        if (errno == EINTR) continue;
        return IpcError::SocketError;
    }
    return IpcError::OK;
}

IpcError write_exact(int fd, const void * buf, size_t len) {
    const char * p = static_cast<const char *>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t w = ::write(fd, p + sent, len - sent);
        if (w > 0) { sent += static_cast<size_t>(w); continue; }
        if (w < 0) {
            if (errno == EINTR) continue;
            return IpcError::SocketError;
        }
    }
    return IpcError::OK;
}

IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id,
                    const void * payload, size_t payload_len) {
    if (payload_len > MAX_FRAME_PAYLOAD) return IpcError::PayloadTooBig;

    FrameHeader hdr {
        /*type=*/   static_cast<uint32_t>(type),
        /*len=*/    static_cast<uint32_t>(payload_len),
        /*req_id=*/ req_id,
    };

    if (payload_len == 0) {
        return write_exact(fd, &hdr, sizeof(hdr));
    }

    iovec iov[2];
    iov[0].iov_base = &hdr;
    iov[0].iov_len  = sizeof(hdr);
    iov[1].iov_base = const_cast<void *>(payload);
    iov[1].iov_len  = payload_len;

    size_t total = sizeof(hdr) + payload_len;
    size_t sent = 0;
    while (sent < total) {
        iovec * cur_iov = iov;
        int n_iov = 2;
        size_t to_skip = sent;
        if (to_skip >= sizeof(hdr)) {
            cur_iov = &iov[1];
            n_iov = 1;
            to_skip -= sizeof(hdr);
            cur_iov[0].iov_base = static_cast<char *>(iov[1].iov_base) + to_skip;
            cur_iov[0].iov_len  = payload_len - to_skip;
        } else if (to_skip > 0) {
            iov[0].iov_base = reinterpret_cast<char *>(&hdr) + to_skip;
            iov[0].iov_len  = sizeof(hdr) - to_skip;
        }

        ssize_t w = ::writev(fd, cur_iov, n_iov);
        if (w > 0) { sent += static_cast<size_t>(w); continue; }
        if (w < 0) {
            if (errno == EINTR) continue;
            return IpcError::SocketError;
        }
    }
    return IpcError::OK;
}

IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id,
                    const std::string & json) {
    return send_frame(fd, type, req_id, json.data(), json.size());
}

IpcError send_frame(int fd, WorkerFrame type, uint32_t req_id,
                    const std::vector<uint8_t> & payload) {
    return send_frame(fd, type, req_id, payload.data(), payload.size());
}

IpcError recv_frame(int fd, FrameHeader * out_hdr,
                    std::vector<uint8_t> * out_payload) {
    if (!out_hdr) return IpcError::ProtocolError;
    IpcError e = read_exact(fd, out_hdr, sizeof(*out_hdr));
    if (e != IpcError::OK) return e;
    if (out_hdr->len > MAX_FRAME_PAYLOAD) return IpcError::PayloadTooBig;
    out_payload->resize(out_hdr->len);
    if (out_hdr->len == 0) return IpcError::OK;
    return read_exact(fd, out_payload->data(), out_hdr->len);
}

std::vector<uint8_t> pack_blob_payload(const std::string & json_meta,
                                       const float * floats,
                                       size_t n_floats) {
    const size_t json_len = json_meta.size();
    const size_t bytes    = n_floats * sizeof(float);
    std::vector<uint8_t> out;
    out.resize(sizeof(uint32_t) + json_len + bytes);

    uint32_t jlen = static_cast<uint32_t>(json_len);
    std::memcpy(out.data(), &jlen, sizeof(jlen));
    if (json_len) {
        std::memcpy(out.data() + sizeof(jlen), json_meta.data(), json_len);
    }
    if (bytes) {
        std::memcpy(out.data() + sizeof(jlen) + json_len, floats, bytes);
    }
    return out;
}

bool unpack_blob_payload(const std::vector<uint8_t> & payload,
                         std::string * out_meta,
                         std::vector<float> * out_floats) {
    if (payload.size() < sizeof(uint32_t)) return false;
    uint32_t jlen = 0;
    std::memcpy(&jlen, payload.data(), sizeof(jlen));
    if (sizeof(jlen) + jlen > payload.size()) return false;
    if (out_meta) {
        out_meta->assign(reinterpret_cast<const char *>(payload.data() + sizeof(jlen)),
                         jlen);
    }
    size_t bytes_off = sizeof(jlen) + jlen;
    size_t bytes     = payload.size() - bytes_off;
    if (bytes % sizeof(float) != 0) return false;
    if (out_floats) {
        out_floats->resize(bytes / sizeof(float));
        if (bytes) {
            std::memcpy(out_floats->data(), payload.data() + bytes_off, bytes);
        }
    }
    return true;
}

std::vector<uint8_t> pack_multi_blob_payload(
    const std::string & json_meta,
    const std::vector<std::string> & blobs) {
    const size_t json_len = json_meta.size();
    size_t total = sizeof(uint32_t) + json_len;
    for (const auto & b : blobs) total += sizeof(uint32_t) + b.size();
    std::vector<uint8_t> out;
    out.resize(total);
    size_t off = 0;
    uint32_t jlen = static_cast<uint32_t>(json_len);
    std::memcpy(out.data() + off, &jlen, sizeof(jlen));
    off += sizeof(jlen);
    if (json_len) {
        std::memcpy(out.data() + off, json_meta.data(), json_len);
        off += json_len;
    }
    for (const auto & b : blobs) {
        uint32_t blen = static_cast<uint32_t>(b.size());
        std::memcpy(out.data() + off, &blen, sizeof(blen));
        off += sizeof(blen);
        if (blen) {
            std::memcpy(out.data() + off, b.data(), blen);
            off += blen;
        }
    }
    return out;
}

bool unpack_multi_blob_payload(
    const std::vector<uint8_t> & payload,
    std::string * out_meta,
    std::vector<std::string> * out_blobs) {
    if (payload.size() < sizeof(uint32_t)) return false;
    uint32_t jlen = 0;
    std::memcpy(&jlen, payload.data(), sizeof(jlen));
    if (sizeof(jlen) + (size_t)jlen > payload.size()) return false;
    if (out_meta) {
        out_meta->assign(reinterpret_cast<const char *>(payload.data() + sizeof(jlen)),
                         jlen);
    }
    if (out_blobs) out_blobs->clear();
    size_t off = sizeof(jlen) + (size_t)jlen;
    while (off < payload.size()) {
        if (off + sizeof(uint32_t) > payload.size()) return false;
        uint32_t blen = 0;
        std::memcpy(&blen, payload.data() + off, sizeof(blen));
        off += sizeof(blen);
        if (off + (size_t)blen > payload.size()) return false;
        if (out_blobs) {
            out_blobs->emplace_back(reinterpret_cast<const char *>(payload.data() + off),
                                    (size_t)blen);
        }
        off += blen;
    }
    return true;
}

pid_t spawn_worker(const char * self_argv0,
                   const std::vector<std::string> & extra_argv,
                   int * out_parent_fd,
                   const std::string & cuda_visible_devices) {
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "spawn_worker: socketpair failed: %s\n", strerror(errno));
        return -1;
    }

    int parent_fd = sv[0];
    int worker_fd = sv[1];
    int flags = ::fcntl(parent_fd, F_GETFD);
    if (flags >= 0) ::fcntl(parent_fd, F_SETFD, flags | FD_CLOEXEC);

    pid_t pid = ::fork();
    if (pid < 0) {
        fprintf(stderr, "spawn_worker: fork failed: %s\n", strerror(errno));
        ::close(sv[0]);
        ::close(sv[1]);
        return -1;
    }

    if (pid == 0) {
        ::close(parent_fd);

        // Pin this worker to a specific GPU before execv (parent is CUDA-free,
        // so the child's primary context lands on the requested device).
        if (!cuda_visible_devices.empty()) {
            ::setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices.c_str(), 1);
        }

        char fd_buf[16];
        std::snprintf(fd_buf, sizeof(fd_buf), "%d", worker_fd);

        std::vector<std::string> owned;
        owned.emplace_back(self_argv0);
        owned.emplace_back("--worker");
        owned.emplace_back(fd_buf);
        for (auto & a : extra_argv) owned.push_back(a);

        std::vector<char *> argv_p;
        argv_p.reserve(owned.size() + 1);
        for (auto & s : owned) argv_p.push_back(s.data());
        argv_p.push_back(nullptr);

        ::execv(self_argv0, argv_p.data());
        std::fprintf(stderr, "spawn_worker child: execv(%s) failed: %s\n",
                     self_argv0, strerror(errno));
        ::_exit(127);
    }

    ::close(worker_fd);
    if (out_parent_fd) *out_parent_fd = parent_fd;
    return pid;
}

} // namespace siglip2
