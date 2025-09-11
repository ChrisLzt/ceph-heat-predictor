#ifndef CEPH_BLK_TIEREDDEVICE_H
#define CEPH_BLK_TIEREDDEVICE_H

#include "BlockDevice.h"

class TieredDevice : public BlockDevice {
public:
    void aio_submit(IOContext *ioc) override;
    void discard_drain() override;
    int collect_metadata(const std::string& prefix, std::map<std::string,std::string> *pm) const override;
    int get_devname(std::string *s) const override;
    int get_devices(std::set<std::string> *ls) const override;

    bool get_thin_utilization(uint64_t *total, uint64_t *avail) const override;

    int read(uint64_t off, uint64_t len, ceph::buffer::list *pbl,
        IOContext *ioc,
        bool buffered) override;
    int aio_read(uint64_t off, uint64_t len, ceph::buffer::list *pbl,
            IOContext *ioc) override;
    int read_random(uint64_t off, uint64_t len, char *buf, bool buffered) override;

    int write(uint64_t off, ceph::buffer::list& bl, bool buffered, int write_hint = WRITE_LIFE_NOT_SET) override;
    int aio_write(uint64_t off, ceph::buffer::list& bl,
            IOContext *ioc,
            bool buffered,
            int write_hint = WRITE_LIFE_NOT_SET) override;
    int flush() override;
    int discard(uint64_t offset, uint64_t len) override;

    // for managing buffered readers/writers
    int invalidate_cache(uint64_t off, uint64_t len) override;
    int open(const std::string& path) override;
    void close() override;
};

#endif // CEPH_BLK_TIEREDDEVICE_H