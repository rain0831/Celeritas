#include "utils.h"

#include <unistd.h>

#include <iostream>

#include "logger.h"

bool has_nans(torch::Tensor values) {
    return torch::isnan(values).any().item<bool>();

}

void process_mem_usage() {
    double vm_usage = 0.0;
    double resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;

    SPDLOG_DEBUG("VM Usage: {}GB. RSS: {}GB", vm_usage / pow(2, 20), resident_set / pow(2, 20));
}

void * memset_wrapper(void *ptr, int value, int64_t num) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < num) {
        curr_bytes = num - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memset((char *) ptr + local_offset, value, curr_bytes);

        local_offset += curr_bytes;
    }

    return ptr;
}

void * memcpy_wrapper(void* dest, const void* src, int64_t count) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memcpy((char *) dest + local_offset, (char *) src + local_offset, curr_bytes);

        local_offset += curr_bytes;
    }

    return dest;
}

int64_t pread_wrapper(int fd, void *buf, int64_t count, int64_t offset) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        if (pread(fd, (char *) buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

int64_t pwrite_wrapper(int fd, const void *buf, int64_t count, int64_t offset) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        if (pwrite(fd, (char *) buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

torch::Tensor transfer_tensor(torch::Tensor input, torch::Device device, at::cuda::CUDAStream *compute_stream, at::cuda::CUDAStream *transfer_stream) {
    if (input.defined()) {
        input = input.pin_memory().to(device, false);

        if (compute_stream != nullptr)
            input.record_stream(*compute_stream);
        if (transfer_stream != nullptr)
            input.record_stream(*transfer_stream);
    }
    return input;
}

std::string tensor_to_string(const torch::Tensor& tensor) {
    std::ostringstream oss;
    if (tensor.scalar_type() == torch::kInt32) {
        auto data = tensor.data_ptr<int32_t>();
        for (int i = 0; i < tensor.numel(); ++i) {
            if (!(oss.str().empty())) {
                oss << " ";
            }
            oss << data[i];
        }
    } else if (tensor.scalar_type() == torch::kInt64) {
        auto data = tensor.data_ptr<int64_t>();
        for (int i = 0; i < tensor.numel(); ++i) {
            if (!(oss.str().empty())) {
                oss << " ";
            }
            oss << data[i];
        }
    } else {
        oss << "Unsupported dtype: " << torch::toString(tensor.scalar_type());
    }
    return oss.str();
}

std::string tensor_to_string(const std::vector<torch::Tensor>& tensors) {
    std::ostringstream oss;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i > 0) {
            oss << "; ";  // 每个张量之间换行
        }
        oss << tensor_to_string(tensors[i]);
    }
    return oss.str();
}