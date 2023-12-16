#include <mutex>
#include "EvoCache.hpp"

template<typename KeyType, typename ValueType>
void EvoCache<KeyType, ValueType>::put(const KeyType& key, const ValueType& value) noexcept {
    auto it = _map.find(key);

    //put if item is not in the cache
    if (it == _map.end()) {

        // check if cache is full, if so, remove the last item
        if (_map.size() >= limit_size) {
            auto i = --_list.end();
            _map.erase(*i);
            _list.erase(i);
        }

        // push to both list and map finally
        _list.push_front(key);
        _map[key] = std::make_pair(value, _list.begin());
    }
}

template<typename KeyType, typename ValueType>
std::optional<ValueType> EvoCache<KeyType, ValueType>::get(const KeyType& key) noexcept {

    auto it = _map.find(key);

    //if item is not in the cache
    if (it == _map.end()) {
        return std::nullopt;
    }

    typename std::list<KeyType>::iterator j = it->second.second;
    if (j != _list.begin()) {

        // item is not at the front
        // we need to move it there

        // remove item from list as first
        _list.erase(j);
        _list.push_front(key);

        // update iterator in map
        j = _list.begin();
        const ValueType &value = it->second.first;
        _map[key] = std::make_pair(value, j);

        // return the value
        return value;
    }
    else {

        // item is already at the front
        // we dont need to move it
        return it->second.first;
    }
}

template void EvoCache<std::string, double>::put(const std::string& key, const double& value) noexcept;
template std::optional<double> EvoCache<std::string, double>::get(const std::string& key) noexcept;