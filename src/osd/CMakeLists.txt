set(osdc_osd_srcs
  ${CMAKE_SOURCE_DIR}/src/osdc/Objecter.cc
  ${CMAKE_SOURCE_DIR}/src/osdc/Striper.cc)

set(HP_ENABLE_STANDARD_SCALER "1" CACHE STRING
  "Heat Predictor online StandardScaler: 0=disabled, 1=enabled")
set_property(CACHE HP_ENABLE_STANDARD_SCALER PROPERTY STRINGS 0 1)
if(NOT HP_ENABLE_STANDARD_SCALER MATCHES "^[01]$")
  message(FATAL_ERROR "HP_ENABLE_STANDARD_SCALER must be 0 or 1")
endif()

set(HP_HEAT_MARGIN_PROFILE "0" CACHE STRING
  "Heat Predictor heat margin profile: 0=current, 1=projected, 2=both")
set_property(CACHE HP_HEAT_MARGIN_PROFILE PROPERTY STRINGS 0 1 2)
if(NOT HP_HEAT_MARGIN_PROFILE MATCHES "^[012]$")
  message(FATAL_ERROR "HP_HEAT_MARGIN_PROFILE must be 0, 1, or 2")
endif()

set_property(SOURCE ObjectHeatPredictor.cc APPEND PROPERTY COMPILE_DEFINITIONS
  HP_ENABLE_STANDARD_SCALER=${HP_ENABLE_STANDARD_SCALER}
  HP_HEAT_MARGIN_PROFILE=${HP_HEAT_MARGIN_PROFILE})

if(WITH_OSD_INSTRUMENT_FUNCTIONS AND CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  add_compile_options(
    -finstrument-functions
    -finstrument-functions-exclude-function-list=_mm_loadu_si128,_mm_cmpeq_epi32,_mm_movemask_epi8)
  set(osd_cyg_functions_src ${CMAKE_SOURCE_DIR}/src/tracing/cyg_profile_functions.c)
endif()

set(osd_srcs
  OSD.cc
  ObjectHeatPredictor.cc
  Watch.cc
  ClassHandler.cc
  PG.cc
  PGLog.cc
  PrimaryLogPG.cc
  ReplicatedBackend.cc
  ECBackend.cc
  ECTransaction.cc
  PGBackend.cc
  OSDCap.cc
  scrubber/pg_scrubber.cc
  scrubber/osd_scrub_sched.cc
  scrubber/PrimaryLogScrub.cc
  scrubber/scrub_machine.cc
  scrubber/ScrubStore.cc
  Watch.cc
  Session.cc
  SnapMapper.cc
  osd_types.cc
  ECUtil.cc
  ExtentCache.cc
  scheduler/OpScheduler.cc
  scheduler/OpSchedulerItem.cc
  scheduler/mClockScheduler.cc
  PeeringState.cc
  PGStateUtils.cc
  recovery_types.cc
  MissingLoc.cc
  osd_perf_counters.cc
  ${CMAKE_SOURCE_DIR}/src/common/TrackedOp.cc
  ${CMAKE_SOURCE_DIR}/src/mgr/OSDPerfMetricTypes.cc
  ${osd_cyg_functions_src}
  ${osdc_osd_srcs}
  osd_tracer.cc)
if(HAS_VTA)
  set_source_files_properties(osdcap.cc
    PROPERTIES COMPILE_FLAGS -fno-var-tracking-assignments)
endif()
add_library(osd STATIC ${osd_srcs})
target_link_libraries(osd
  PUBLIC dmclock::dmclock Boost::MPL
  PRIVATE os heap_profiler cpu_profiler fmt::fmt ${CMAKE_DL_LIBS})
if(WITH_LTTNG)
  add_dependencies(osd osd-tp pg-tp)
endif()
if(WITH_EVENTTRACE)
  add_dependencies(osd eventtrace_tp)
endif()
if(WITH_OSD_INSTRUMENT_FUNCTIONS)
  add_dependencies(osd cyg_profile_tp)
endif()

# libcls_* are runtime dependencies
add_dependencies(osd cls_journal cls_hello cls_lock cls_log cls_numops
  cls_refcount cls_timeindex cls_user cls_version cls_cas cls_cmpomap)
if(WITH_CEPHFS)
  add_dependencies(osd cls_cephfs)
endif()
if(WITH_RBD)
  add_dependencies(osd cls_rbd)
endif()
if(WITH_RADOSGW)
  add_dependencies(osd cls_otp cls_rgw cls_queue cls_rgw_gc cls_2pc_queue cls_fifo)
endif()
