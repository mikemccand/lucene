/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.util;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.tests.util.LuceneTestCase.Monster;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static org.junit.Assert.*;

public class TestOffHeapBytesRefHash extends LuceneTestCase {

  private OffHeapBytesRefHash h;

  @Before
  public void setup() throws Exception {
    // Put temp file mapping under the test’s temp dir
    Path dir = createTempDir("offheap-bytesrefhash");
    h = new OffHeapBytesRefHash(16, dir);
  }

  @After
  public void tearDown() throws Exception {
    if (h != null) {
      h.close();
    }
    super.tearDown();
  }

  @Test
  public void testEmpty() {
    assertEquals(0, h.size());
    assertEquals(-1, h.find(new BytesRef()));
  }

  @Test
  public void testAddAndFindUniq() {
    int N = atLeast(1000);
    List<BytesRef> refs = new ArrayList<>(N);
    for (int i = 0; i < N; i++) {
      refs.add(randomBytesRef(i % 64));
    }
    for (int i = 0; i < N; i++) {
      int ord = h.add(refs.get(i));
      assertEquals(i, ord);
      assertEquals(i, h.find(refs.get(i)));
    }
    assertEquals(N, h.size());
  }

  @Test
  public void testDuplicatesReturnNegativeOrd() {
    BytesRef a = new BytesRef(new byte[]{1,2,3,4});
    int ord = h.add(a);
    assertEquals(0, ord);
    int dup = h.add(a);
    assertEquals(-(ord + 1), dup);
    assertEquals(ord, h.find(a));
  }

  @Test
  public void testGetRoundTrip() {
    int N = atLeast(500);
    List<BytesRef> refs = new ArrayList<>(N);
    for (int i = 0; i < N; i++) {
      refs.add(randomBytesRef(1 + random().nextInt(200)));
      h.add(refs.get(i));
    }
    BytesRef scratch = new BytesRef();
    for (int i = 0; i < N; i++) {
      h.get(i, scratch);
      assertArrayEquals(copyBytes(refs.get(i)), copyBytes(scratch));
    }
  }

  @Test
  public void testCompactContainsAllOrds() {
    for (int i = 0; i < 257; i++) {
      h.add(new BytesRef(("k" + i).getBytes(StandardCharsets.UTF_8)));
    }
    int[] comp = h.compact();
    assertEquals(h.size(), comp.length);
    boolean[] seen = new boolean[h.size()];
    for (int v : comp) {
      assertTrue(v >= 0 && v < h.size());
      assertFalse(seen[v]);
      seen[v] = true;
    }
  }

  @Test
  public void testClearAndReuse() {
    h.add(new BytesRef("abc".getBytes(StandardCharsets.UTF_8)));
    h.add(new BytesRef("def".getBytes(StandardCharsets.UTF_8)));
    assertEquals(2, h.size());
    h.clear();
    assertEquals(0, h.size());
    assertEquals(-1, h.find(new BytesRef("abc".getBytes(StandardCharsets.UTF_8))));
    assertTrue(h.add(new BytesRef("abc".getBytes(StandardCharsets.UTF_8))) >= 0);
  }

  @Test
  public void testLengthHeaderBoundaries() {
    // 1-byte header (<128), 2-byte header (>=128), and the max 32767
    for (int len : new int[]{0, 1, 2, 63, 127, 128, 1024, 8191, 32767}) {
      byte[] b = new byte[len];
      Arrays.fill(b, (byte) 7);
      int ord = h.add(new BytesRef(b));
      assertTrue(ord >= 0);
      assertEquals(ord, h.find(new BytesRef(b)));
    }
  }

  @Test
  public void testTooLongThrows() {
    byte[] b = new byte[32768];
    expectThrows(OffHeapBytesRefHash.MaxBytesLengthExceededException.class, () -> {
      h.add(new BytesRef(b));
    });
  }

  @Test
  public void testRandomAddsWithCollisions() {
    int N = atLeast(2000);
    Set<BytesRef> seen = new HashSet<>();
    for (int i = 0; i < N; i++) {
      BytesRef br = randomBytesRef(random().nextInt(256));
      int added = h.add(br);
      boolean first = seen.add(new BytesRef(br.bytes, br.offset, br.length));
      if (first) {
        assertTrue(added >= 0);
      } else {
        assertTrue("expected duplicate negative ord", added < 0);
      }
    }
    // Verify a random subset via get()
    BytesRef scratch = new BytesRef();
    int toCheck = Math.min(h.size(), 100);
    int[] ords = randomUniqueInts(random(), toCheck, 0, h.size());
    for (int ord : ords) {
      h.get(ord, scratch);
      assertEquals(ord, h.find(scratch));
    }
  }

  @Test
  public void testRamBytesUsedIsStable() {
    long before = h.ramBytesUsed();
    for (int i = 0; i < 512; i++) {
      h.add(randomBytesRef(1 + (i % 64)));
    }
    long after = h.ramBytesUsed();
    assertTrue(after >= before);
  }

  // ------------------------- MONSTER -------------------------

  /**
   * Writes more than 2 GiB of total BytesRef contents into the off-heap file
   * and verifies lookups and random spot-check retrievals. Annotated as a Monster
   * test so it only runs under -Dtests.monster=true.
   */
  @Test
  @Category(Monster.class)
  public void test2GBPlusTotalBytes() throws Exception {
    // Use a larger initial hash to reduce rehash churn
    try (OffHeapBytesRefHash big = new OffHeapBytesRefHash(1 << 18, createTempDir("offheap-2gb"))) {

      final long TARGET = (2L << 30) + (64L << 20); // 2 GiB + 64 MiB buffer
      final int FIXED_LEN = 32000;                  // < 32768, forces 2-byte header
      final int HEADER = 2;                         // because len >= 128
      final int STEP_BYTES = FIXED_LEN + HEADER;

      // To validate duplicates, insert one "dup" upfront and again later
      byte[] dup = new byte[FIXED_LEN];
      Arrays.fill(dup, (byte) 0x5A);
      int dupOrd = big.add(new BytesRef(dup));
      assertTrue(dupOrd >= 0);
      assertEquals(dupOrd, big.find(new BytesRef(dup)));

      // Reservoir sample a small set of ord->bytes for later verification
      final int SAMPLE = 512;
      final int SAMPLE_MASK = SAMPLE - 1;
      BytesRef[] sampleVals = new BytesRef[SAMPLE];
      int[] sampleOrds = new int[SAMPLE];
      int sampleCount = 0;

      // Advance until actual written bytes exceed target
      int i = 0;
      while (big.offHeapBytesUsed() < TARGET) {
        byte[] buf = new byte[FIXED_LEN];
        // Make content deterministic but varying, not compressible pattern
        int seed = i * 2654435761L != 0 ? i : 1;
        for (int k = 0; k < FIXED_LEN; k++) {
          // xorshift-ish pattern
          seed ^= (seed << 13);
          seed ^= (seed >>> 17);
          seed ^= (seed << 5);
          buf[k] = (byte) seed;
        }
        BytesRef br = new BytesRef(buf);
        int ord = big.add(br);
        if ((i & SAMPLE_MASK) == 0) { // light sampling
          sampleVals[sampleCount & SAMPLE_MASK] = BytesRef.deepCopyOf(br);
          sampleOrds[sampleCount & SAMPLE_MASK] = ord >= 0 ? ord : -(ord + 1);
          sampleCount++;
        }
        i++;

        // Periodically re-add the dup to ensure duplicates are still detected
        if ((i % 10000) == 0) {
          int d = big.add(new BytesRef(dup));
          assertEquals(-(dupOrd + 1), d);
          assertEquals(dupOrd, big.find(new BytesRef(dup)));
        }

        // Optional: sanity progress assertion to avoid infinite loops
        if (i % 20000 == 0) {
          assertTrue("offHeapBytesUsed should advance", big.offHeapBytesUsed() >= (long) i * STEP_BYTES - (10L * STEP_BYTES));
        }
      }

      // We truly crossed 2 GiB of actual written content
      assertTrue("used=" + big.offHeapBytesUsed(), big.offHeapBytesUsed() > (2L << 30));

      // Spot check sample retrieval and find()
      BytesRef scratch = new BytesRef();
      int checks = Math.min(SAMPLE, sampleCount);
      for (int s = 0; s < checks; s++) {
        int ord = sampleOrds[s];
        big.get(ord, scratch);
        assertArrayEquals(copyBytes(sampleVals[s]), copyBytes(scratch));
        assertEquals(ord, big.find(sampleVals[s]));
      }

      // Compact should include all ords and be a permutation of [0..size)
      int[] comp = big.compact();
      assertEquals(big.size(), comp.length);
      boolean[] seen = new boolean[big.size()];
      for (int v : comp) {
        assertTrue(v >= 0 && v < big.size());
        assertFalse(seen[v]);
        seen[v] = true;
      }
    }
  }

  // ------------------------- helpers -------------------------

  private static byte[] copyBytes(BytesRef br) {
    return Arrays.copyOfRange(br.bytes, br.offset, br.offset + br.length);
  }

  private BytesRef randomBytesRef(int maxLenExclusive) {
    int len = maxLenExclusive <= 0 ? 0 : random().nextInt(maxLenExclusive);
    // keep below MAX_VALUE_LENGTH
    len = Math.min(len, 32767);
    byte[] b = new byte[len];
    random().nextBytes(b);
    return new BytesRef(b);
  }

  /** Local replacement for TestUtil.randomUniqueInts(...) */
  private static int[] randomUniqueInts(Random r, int count, int minInclusive, int maxExclusive) {
    if (count < 0 || minInclusive > maxExclusive) throw new IllegalArgumentException();
    int span = maxExclusive - minInclusive;
    if (count > span) throw new IllegalArgumentException("count > range");
    // small count → reservoir-style sampling using a HashSet
    if (count <= 1024) {
      HashSet<Integer> set = new HashSet<>(count * 2);
      while (set.size() < count) {
        set.add(minInclusive + r.nextInt(span));
      }
      int[] out = new int[count];
      int i = 0;
      for (int v : set) out[i++] = v;
      return out;
    }
    // large count → shuffle a list of the range and take first N
    ArrayList<Integer> list = new ArrayList<>(span);
    for (int v = minInclusive; v < maxExclusive; v++) list.add(v);
    Collections.shuffle(list, r);
    int[] out = new int[count];
    for (int i = 0; i < count; i++) out[i] = list.get(i);
    return out;
  }
}
