package com.hora.app;

import com.hora.app.ANNIndex;

public class HNSWIndex extends ANNIndex {
    private String index_key;

    public HNSWIndex(int dimension) {
        index_key = "hnsw";
        new_bf_index(index_key, dimension);
    }
}
