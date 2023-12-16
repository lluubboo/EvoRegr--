#pragma once
#include <stdexcept>
#include <list>
#include <utility>
#include <string>
#include <optional>
#include <sparsehash/dense_hash_map>

template<typename KeyType, typename ValueType>
class EvoCache {

    std::list<KeyType> _list;
    google::dense_hash_map<KeyType, std::pair<ValueType, typename std::list<KeyType>::iterator>> _map;
    size_t limit_size; 

public:
    
    EvoCache(size_t size) :
        _list(),
        _map(),
        limit_size(size)
    {
        if (size == 0) {
            throw std::invalid_argument("Cache size must be greater than 0");
        }

        // required actions to use dense_hash_map
        // key-value that is never used for legitimate hash-map entries
        _map.set_empty_key("empty");

        //  Increases the bucket count to hold at least n items.
        _map.resize(size);
    }

    ~EvoCache(){};	

    void put(const KeyType& key, const ValueType& value) noexcept;
    std::optional<ValueType> get(const KeyType& key) noexcept;
};