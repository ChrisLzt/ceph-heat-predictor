package Vdb;

import java.lang.reflect.Field;
import java.util.Random;
import sun.misc.Unsafe;

public final class FractionalXferTest {
  private static FwgEntry entry(double[] histogram) throws Exception {
    Field f = Unsafe.class.getDeclaredField("theUnsafe");
    f.setAccessible(true);
    Unsafe unsafe = (Unsafe) f.get(null);
    FwgEntry result = (FwgEntry) unsafe.allocateInstance(FwgEntry.class);
    result.xfersizes = histogram;
    Field rng = FwgEntry.class.getDeclaredField("xfer_size_randomizer");
    rng.setAccessible(true);
    rng.set(result, new Random(0));
    return result;
  }

  private static void check(double[] histogram, int draws) throws Exception {
    FwgEntry e = entry(histogram);
    long[] counts = new long[histogram.length / 2];
    for (int n = 0; n < draws; n++) {
      int size = e.getXferSize();
      boolean found = false;
      for (int i = 0; i < counts.length; i++) {
        if (size == (int) histogram[2 * i]) {
          counts[i]++;
          found = true;
          break;
        }
      }
      if (!found) throw new AssertionError("unexpected transfer size: " + size);
    }
    for (int i = 0; i < counts.length; i++) {
      double probability = histogram[2 * i + 1] / 100.0;
      double expected = draws * probability;
      double tolerance = 8 * Math.sqrt(expected * (1 - probability)) + 2;
      if (Math.abs(counts[i] - expected) > tolerance)
        throw new AssertionError("distribution mismatch at " + i);
    }
    System.out.println("PASS histogram " + java.util.Arrays.toString(histogram)
                       + " counts=" + java.util.Arrays.toString(counts));
  }

  public static void main(String[] args) throws Exception {
    check(new double[]{4096, 0.25, 8192, 33.125, 16384, 66.625}, 3000000);
    check(new double[]{4096, 50, 8192, 50}, 3000000);
    check(new double[]{4096, 99.99, 8192, 0.01}, 3000000);
    FwgEntry single = entry(new double[]{4096});
    for (int i = 0; i < 100; i++)
      if (single.getXferSize() != 4096) throw new AssertionError("single size");
    System.out.println("PASS singleton; actual FwgEntry.getXferSize invoked");
  }
}
