#include <mutex>
#include "EvoCache.hpp"

template<typename KeyType, typename ValueType>
void EvoCache<KeyType, ValueType>::put(const KeyType key, const ValueType value) noexcept {
    std::unique_lock lock(_mutex);
    {
        auto it = _cache.find(key);

        if (it == _cache.end()) { // map does not contain key
            
            if (_keys.size() == limit_size) { // cache is full, erase first n entries
                for (size_t i = 0; i < limit_size / 3; i++) {
                    _cache.erase(_keys.front());
                    _keys.pop_front();
                }
            }

            // after erasing if needed, insert new key-value pair
            _cache[key] = value;
            _keys.push_back(key);
        }
    }
}

template<typename KeyType, typename ValueType>
std::optional<ValueType> EvoCache<KeyType, ValueType>::get(const KeyType key) const noexcept {
    std::shared_lock lock(_mutex);
    {
        auto it = _cache.find(key);
        if (it != _cache.end()) {
            return it->second;
        }
        else {
            return std::nullopt;
        }
    }
}

template void EvoCache<std::string, double>::put(const std::string key, const double value) noexcept;
template std::optional<double> EvoCache<std::string, double>::get(const std::string key) const noexcept;