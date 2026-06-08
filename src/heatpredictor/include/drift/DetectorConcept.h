#ifndef DETECTOR_CONCEPT_H
#define DETECTOR_CONCEPT_H

#include <type_traits>
#include <utility> // for std::move, std::declval

// ==========================================
// 1. 实现 IsDetector 概念检查 (C++17 style)
// ==========================================

// 辅助模板：用于检查 "update(double)" 和 "drift_detected"
template <typename T, typename = void>
struct is_detector_impl : std::false_type {};

template <typename T>
struct is_detector_impl<T, std::void_t<
    // SFINAE 检查表达式是否合法：
    decltype(std::declval<T>().update(1.0)),                // 检查是否有 update(double)
    decltype(std::declval<T&>().drift_detected = true)      // 检查 drift_detected 是否存在且可赋值
>> : std::integral_constant<bool,
    // 进一步检查类型约束：
    std::is_same_v<void, decltype(std::declval<T>().update(1.0))> &&          // update 返回 void
    std::is_convertible_v<decltype(std::declval<T>().drift_detected), bool> && // drift_detected 可转为 bool
    std::is_move_constructible_v<T> &&                                        // 是可移动构造的
    std::is_move_assignable_v<T>                                              // 是可移动赋值的
> {};

// 定义便利的变量模板 (类似 C++20 的 concept 用法)
template <typename T>
inline constexpr bool IsDetector_v = is_detector_impl<T>::value;


// ==========================================
// 2. 实现 DetectorFactory 类
// ==========================================

// C++17 支持 "auto ... Args" 非类型模板参数
template <typename D, auto ... Args>
struct DetectorFactory {
    // 1. 概念检查
    static_assert(IsDetector_v<D>, "Template parameter D must satisfy IsDetector concept");

    using DetectorType = D;

    static D create() {
        // 关键点：参数包展开 (Parameter Pack Expansion)
        // 语法 (expression ... ) 会对包中的每个元素应用该表达式
        // 这里我们将 Args 中的每个值都强转为 double 并除以 1000.0
        return D( (static_cast<double>(Args) / 1000.0)... );
    }
};


// ==========================================
// 3. 实现 IsDetectorFactory 概念检查
// ==========================================

template <typename F, typename = void>
struct is_detector_factory_impl : std::false_type {};

template <typename F>
struct is_detector_factory_impl<F, std::void_t<
    typename F::DetectorType,   // 检查是否有 DetectorType 类型别名
    decltype(F::create())       // 检查是否有静态 create() 方法
>> : std::integral_constant<bool,
    // 检查 create() 的返回值是否是 DetectorType
    std::is_same_v<typename F::DetectorType, decltype(F::create())> &&
    // 检查生产出来的类型是否满足 IsDetector
    IsDetector_v<typename F::DetectorType>
> {};

template <typename F>
inline constexpr bool IsDetectorFactory_v = is_detector_factory_impl<F>::value;

#endif // DETECTOR_CONCEPT_H