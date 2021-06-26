class ANNIndexManager {

    public static native void init();

    public static native void new_bf_index(String name, int dimension);

    public static native void add(String name, float[] features, int features_idx);

    public static native void build(String name, String mt);

    public static native int[] search(String name, int k, float[] features);

    static {
        System.loadLibrary("hora");
    }
}