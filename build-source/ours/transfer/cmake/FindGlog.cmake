# Compatibility for legacy find_package(Glog) with lowercase distro configs.
find_package(glog CONFIG REQUIRED)
set(Glog_FOUND TRUE)
