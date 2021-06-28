package com.hora.app;

public class ANNIndex {
    public static native void new_bf_index(String name, int dimension);

    public static native void add(String name, float[] features, int features_idx);

    public static native void build(String name, String mt);

    public static native int[] search(String name, int k, float[] features);

    public static native void load(String name, String file_path);

    public static native void dump(String name, String file_path);

    static {
        System.loadLibrary("hora");
    }
}
