#include "os/ObjectStoreAccess.h"
#include "osd/ObjectHeatPredictor.h"
#include "common/ceph_context.h"
#include "common/hobject.h"
#include <cassert>
#include <pthread.h>
#include <thread>
#include <vector>
static thread_local unsigned reads=0;
extern "C" int __real_pthread_rwlock_rdlock(pthread_rwlock_t*);
extern "C" int __wrap_pthread_rwlock_rdlock(pthread_rwlock_t* p){++reads;return __real_pthread_rwlock_rdlock(p);}
int main(){
 CephContext context(CEPH_ENTITY_TYPE_OSD);
 ObjectStoreAccess<hobject_t> observer;ObjectHeatPredictor hp;
 auto gate=observer.observation_gate();hp.init(&context,0,gate);
 std::atomic<unsigned> callbacks{0};
 observer.set([&](const hobject_t& o,HpAccessType k,uint64_t n){++callbacks;hp.observe(o,k,n);});
 hobject_t object(object_t("gate-test"),"",CEPH_NOSNAP,42,1,"");
 auto command=[&](const char* cmd){assert(hp.handle_command(cmd,{},nullptr));};
 auto off_check=[&]{auto locks=reads,called=callbacks.load();for(int i=0;i<1000;++i)observer.notify(object,HpAccessType::Read,1);assert(reads==locks);assert(callbacks==called);assert(!gate->load());};
 off_check();command("object_hp reset");off_check();
 command("object_hp enable");assert(gate->load());observer.notify(object,HpAccessType::Read,1);assert(callbacks==1);
 command("object_hp reset");assert(gate->load());observer.notify(object,HpAccessType::Write,1);assert(callbacks==2);
 command("object_hp disable");off_check();
 std::atomic<bool> stop{false};std::vector<std::thread> workers;
 for(int i=0;i<4;++i)workers.emplace_back([&]{while(!stop.load())observer.notify(object,HpAccessType::Read,1);});
 for(int i=0;i<20;++i){command("object_hp enable");command("object_hp reset");command("object_hp disable");}
 stop=true;for(auto& t:workers)t.join();off_check();
 command("object_hp enable");observer.clear();hp.shutdown();assert(!gate->load());observer.notify(object,HpAccessType::Read,1);
}
