package com.hora.app;

import static org.junit.Assert.assertTrue;

import org.junit.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;
public class BruteForceIndexTest 
{
    /**
     * Rigorous Test :-)
     */
    @Test
    public void shouldAnswerWithTrue()
    {
        final int dimension = 2;
        final float variance = 2.0f;
        Random fRandom = new Random();
        String index_name = "demo";

        BruteForceIndex idx = new BruteForceIndex(dimension);
        idx.init();

        List<float[]> tmp = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            for (int p = 0; p < 10; p++) {
                float[] features = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    features[j] = getGaussian(fRandom, (float) (i * 10), variance);
                }
                idx.add(index_name, features, i * 10 + p);
                tmp.add(features);
            }
        }
        idx.build(index_name, "euclidean");

        int search_index = fRandom.nextInt(tmp.size());
        int[] result = idx.search(index_name, 10, tmp.get(search_index));
        System.out.println(Arrays.toString(result));
        assertTrue( true );
    }

    private static float getGaussian(Random fRandom, float aMean, float variance) {
        float r = (float) fRandom.nextGaussian();
        return aMean + r * variance;
    }
}
