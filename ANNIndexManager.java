import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;

class ANNIndexManager {

    private static native void init();

    private static native void new_bf_index(String name, int dimension);

    private static native void add(String name, float[] features, int features_idx);

    private static native void build(String name, String mt);

    private static native int[] search(String name, int k, float[] features);

    static {
        System.loadLibrary("hora");
    }

    public static void main(String[] args) {
        // here is a demo
        final int dimension = 2;
        final float variance = 2.0f;
        Random fRandom = new Random();
        String index_name = "demo";

        init();
        new_bf_index(index_name, dimension);
        List<float[]> tmp = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            for (int p = 0; p < 10; p++) {
                float[] features = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    features[j] = getGaussian(fRandom, (float) (i * 10), variance);
                }
                add(index_name, features, i * 10 + p);
                tmp.add(features);
            }
        }
        build(index_name, "euclidean");

        int search_index = fRandom.nextInt(tmp.size());
        int[] result = search(index_name, 10, tmp.get(search_index));
        System.out.println(Arrays.toString(result));
    }

    private static float getGaussian(Random fRandom, float aMean, float variance) {
        float r = (float) fRandom.nextGaussian();
        return aMean + r * variance;
    }
}