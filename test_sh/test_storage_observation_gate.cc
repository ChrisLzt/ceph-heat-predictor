#include "os/ObjectStoreAccess.h"
#include <cassert>
#include <atomic>
#include <thread>
#include <future>
#include <pthread.h>
static thread_local unsigned reads=0;
extern "C" int __real_pthread_rwlock_rdlock(pthread_rwlock_t*);
extern "C" int __wrap_pthread_rwlock_rdlock(pthread_rwlock_t* p){++reads;return __real_pthread_rwlock_rdlock(p);}
int main(){
 ObjectStoreAccess<int> observer;
 std::atomic<unsigned> callbacks{0};
 observer.set([&](const int&,HpAccessType,uint64_t){++callbacks;});
 auto gate=observer.observation_gate(); gate->store(false);
 const auto before=reads;
 for(int i=0;i<1000;++i)observer.notify(1,HpAccessType::Read,1);
 assert(reads==before); assert(callbacks==0);
 gate->store(true);observer.notify(1,HpAccessType::Read,1);assert(callbacks==1);
 std::promise<void> entered,release;auto ready=release.get_future().share();
 observer.set([&](const int&,HpAccessType,uint64_t){entered.set_value();ready.wait();});
 std::thread io([&]{observer.notify(1,HpAccessType::Write,1);});entered.get_future().wait();
 gate->store(false);const auto off_reads=reads;observer.notify(1,HpAccessType::Read,1);assert(reads==off_reads);
 auto clear=std::async(std::launch::async,[&]{observer.clear();});
 assert(clear.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout);
 release.set_value();io.join();clear.get();gate->store(true);observer.notify(1,HpAccessType::Read,1);
 {ObjectStoreAccess<int> temporary;gate=temporary.observation_gate();}gate->store(false); // lifetime independent
}
