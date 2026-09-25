package Vdb;

import java.util.Random;

public final class FractionalSampler {
  public static int sample(Random random, double[] histogram) {
    if (histogram.length == 1) return (int) histogram[0];
    double percentile = random.nextDouble() * 100.0;
    double cumulative = 0.0;
    for (int i = 0; i < histogram.length; i += 2) {
      cumulative += histogram[i + 1];
      if (percentile < cumulative) return (int) histogram[i];
    }
    if (Math.abs(cumulative - 100.0) > 0.0000001)
      throw new IllegalArgumentException("transfer histogram does not sum to 100");
    return (int) histogram[histogram.length - 2];
  }
}
