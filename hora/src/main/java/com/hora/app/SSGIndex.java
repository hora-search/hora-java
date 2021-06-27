package com.hora.app;
import com.hora.app.ANNIndex;

public class SSGIndex extends ANNIndex {
    private String index_key;
    public SSGIndex(int dimension) {
        index_key = "bf";
        new_bf_index(index_key, dimension);
    }
}
