package com.hora.app;

import static org.junit.Assert.assertTrue;

import org.junit.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import javafx.util.Pair;

public class ANNIndexTest {
    /**
     * Rigorous Test :-)
     */
    private static Logger log = Logger.getLogger(ANNIndexTest.class);

    @Test
    public void shouldAnswerWithTrue() {
        BasicConfigurator.configure();

        final int dimension = 2;
        final float variance = 2.0f;
        Random fRandom = new Random();
        // String index_name = "hello";

        BruteForceIndex bruteforce_idx = new BruteForceIndex(dimension);
        HNSWIndex hnsw_idx = new HNSWIndex(dimension);
        SSGIndex ssg_idx = new SSGIndex(dimension);

        ArrayList<Pair<ANNIndex, String>> idx_list = new ArrayList<Pair<ANNIndex, String>>() {
            {
                add(new Pair<ANNIndex, String>(bruteforce_idx, "bf"));
                add(new Pair<ANNIndex, String>(hnsw_idx, "hnsw"));
                add(new Pair<ANNIndex, String>(ssg_idx, "ssg"));
            }
        };

        for (Pair<ANNIndex, String> idx_pair : idx_list) {
            ANNIndex idx = idx_pair.getKey();
            String index_name = idx_pair.getValue();
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
            log.info("hahah" + Arrays.toString(result));
        }
        demo();
        assertTrue(true);
    }

    public void demo() {
        final int dimension = 2;
        final float variance = 2.0f;
        Random fRandom = new Random();
        // String index_name = "hello";

        BruteForceIndex bruteforce_idx = new BruteForceIndex(dimension);

        List<float[]> tmp = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            for (int p = 0; p < 10; p++) {
                float[] features = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    features[j] = getGaussian(fRandom, (float) (i * 10), variance);
                }
                bruteforce_idx.add("demo", features, i * 10 + p);
                tmp.add(features);
            }
        }
        bruteforce_idx.build("demo", "euclidean");

        int search_index = fRandom.nextInt(tmp.size());
        int[] result = bruteforce_idx.search("demo", 10, tmp.get(search_index));
        log.info("demo bruteforce_idx" + Arrays.toString(result));

    }

    private static float getGaussian(Random fRandom, float aMean, float variance) {
        float r = (float) fRandom.nextGaussian();
        return aMean + r * variance;
    }
}
