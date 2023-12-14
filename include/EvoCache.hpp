#pragma once
#include <shared_mutex>
#include <unordered_map>
#include <stdexcept>
#include <list>
#include <string>
#include <optional>

template<typename KeyType, typename ValueType>
class EvoCache {

    mutable std::shared_mutex _mutex;

    std::list<KeyType> _keys; //keep track of order of insertion
    std::unordered_map<KeyType, ValueType> _cache; // cache

    size_t limit_size; // max cache size

public:
    
    EvoCache(size_t size) :
        _mutex(),
        _keys(),
        _cache(),
        limit_size(size)
    {
        if (size == 0) {
            throw std::invalid_argument("Cache size must be greater than 0");
        }
        
        _cache.reserve(limit_size);
    }

    void put(const KeyType key, const ValueType value) noexcept;
    std::optional<ValueType> get(const KeyType key) const noexcept;
};