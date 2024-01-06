#pragma once
#include <stdexcept>
#include <list>
#include <utility>
#include <string>
#include <optional>
#include <unordered_map>

template<typename KeyType, typename ValueType>
class LRUCache {

    std::list<KeyType> _list;
    std::unordered_map<KeyType, std::pair<ValueType, typename std::list<KeyType>::iterator>> _map;
    size_t _limit_size;

public:

    /**
     * @brief LRUCache class represents a Least Recently Used (LRU) cache.
     * 
     * The LRUCache class is used to store key-value pairs with a fixed size limit.
     * When the cache is full and a new key-value pair is added, the least recently used
     * item is evicted from the cache to make room for the new item.
     * 
     * @param size The maximum number of key-value pairs that the cache can hold.
     */
    LRUCache(size_t size) : _list(), _map(), _limit_size(size) {
        if (_limit_size == 0) {
            throw std::invalid_argument("Cache size must be greater than 0");
        }
        _map.reserve(_limit_size);
    };

    /**
     * Inserts a key-value pair into the cache.
     * If the key already exists in the cache, the value is updated.
     * If the cache is full, the least recently used item is removed.
     *
     * @param key The key of the item to be inserted.
     * @param value The value of the item to be inserted.
     */
    void put(const KeyType& key, const ValueType& value) noexcept {
        auto it = _map.find(key);

        //put if item is not in the cache

        if (it == _map.end()) { //if so....

            // check if cache is full, if so, remove the last item
            if (_map.size() >= _limit_size) {
                auto i = --_list.end();
                _map.erase(*i);
                _list.erase(i);
            }

            // push to both list and map
            _list.push_front(key);
            _map[key] = std::make_pair(value, _list.begin());
        }
    };

    /**
     * Retrieves the value associated with the specified key from the cache.
     * 
     * @param key The key to look up in the cache.
     * @return An optional value that contains the retrieved value if it exists in the cache, or an empty optional if the key is not found.
     */
    std::optional<ValueType> get(const KeyType& key) noexcept {

        auto it = _map.find(key);

        //if item is not in the cache
        if (it == _map.end()) {
            return std::nullopt;
        }

        typename std::list<KeyType>::iterator j = it->second.second; //gets the list iterator from the map

        if (j != _list.begin()) {

            // item is not at the front
            // we need to move it there

            // remove item from list as first
            _list.erase(j);
            _list.push_front(key);

            // update iterator in map
            j = _list.begin();

            const ValueType& value = it->second.first;
            _map[key] = std::make_pair(value, j);

            // return the value
            return value;
        }
        else {

            // item is already at the front
            // we dont need to move it

            return it->second.first; //return the value
        }
    };


    /**
     * @brief Clears the LRUCache by removing all elements from the cache.
     */
    void clear() {
        _list.clear();
        _map.clear();
    };

    /**
     * @brief Checks if the LRUCache is empty.
     * 
     * @return true if the cache is empty, false otherwise.
     */
    bool empty() const {
        return _map.empty();
    };

    /**
     * Returns the number of elements in the LRUCache.
     *
     * @return The number of elements in the LRUCache.
     */
    size_t size() const {
        return _map.size();
    };

    /**
     * Returns the maximum size of the LRUCache.
     *
     * @return The maximum size of the LRUCache.
     */
    size_t limit_size() const {
        return _limit_size;
    };
};