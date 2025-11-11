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

import java.io.Closeable;
import java.io.RandomAccessFile;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;

/**
 * Off-heap version of a hash-map-like structure optimized for BytesRef, modeled after Lucene's
 * BytesRefHash but uses long[] for bytesStart (address into an off-heap mapped file), stores
 * data in a single growable memory-mapped temp file, preserves int[] ids (so ~2.1B entries), with
 * high bits of the hash cached in ids, uses Lucene utilities (ArrayUtil, StringHelper, BytesRef).
 *
 * <p>API is intentionally minimal: add/find/get/compact/clear/close/ramBytesUsed/size.
 *
 * <p>Target: JDK 25 (FFM API) and Lucene 10.3.1+.
 */
public final class OffHeapBytesRefHash implements Accountable, Closeable {

  // --- hashing table ---

  private int hashSize;
  private int hashHalfSize;
  private int hashMask;

  /** Mask for the high bits of the hashCode kept inside ids[]. */
  private int highMask;

  /** -1 for empty; otherwise stores (ord | (hashCode & highMask)). */
  private int[] ids;

  /** Offsets into off-heap storage (pointing at the vint length header). */
  private long[] bytesStart;

  /** Number of unique entries. */
  private int count;

  // --- off-heap storage ---

  private final OffHeapFile pool;

  /** Max allowed length for a single BytesRef payload. */
  private static final int MAX_VALUE_LENGTH = 32767; // 0x7FFF

  /** Default initial hash size (power of two). */
  private static final int INIT_HASH_SIZE = 16;

  /** Create with default capacity. */
  public OffHeapBytesRefHash() throws IOException {
    this(INIT_HASH_SIZE, null);
  }

  /** Create with provided capacity. */
  public OffHeapBytesRefHash(int initialHashSize) throws IOException {
    this(initialHashSize, null);
  }

  /** Create with default capacity in provided temp directory. */
  public OffHeapBytesRefHash(Path tmpDir) throws IOException {
    this(INIT_HASH_SIZE, tmpDir);
  }

  /**
   * Create with an initial hash capacity (which will be rounded up to power of two),
   * temp file in given dir or default (null).
   */
  public OffHeapBytesRefHash(int initialHashSize, Path tmpDir) throws IOException {
    if (initialHashSize < 2) {
      throw new IllegalArgumentException("initialHashSize must be >= 2");
    }
    // round up to next power of two
    int size = 1;
    while (size < initialHashSize) size <<= 1;

    this.hashSize = size;
    this.hashHalfSize = size >>> 1;
    this.hashMask = size - 1;
    this.highMask = ~hashMask;

    this.ids = new int[size];
    Arrays.fill(ids, -1);

    this.bytesStart = new long[16]; // grows on demand
    this.pool = new OffHeapFile(tmpDir);
  }

  /** Number of unique entries. */
  public int size() {
    return count;
  }

  /** Clears the hash table and off-heap storage (but keeps allocated arrays/mapping). */
  public void clear() {
    Arrays.fill(ids, -1);
    hashHalfSize = hashSize >>> 1;
    count = 0;
    pool.clear();
  }

  /**
   * Add the given BytesRef, returning its ordinal if new, or a negative value -(ord+1) if it
   * already existed (same contract as BytesRefHash).
   */
  public int add(BytesRef bytes) {
    if (bytes.length > MAX_VALUE_LENGTH) {
      throw new MaxBytesLengthExceededException("length=" + bytes.length + " >= " + MAX_VALUE_LENGTH);
    }
    //System.out.println("ADD: " + bytes.length);
    int hashCode = doHash(bytes);
    //System.out.println("  hashCode=" + hashCode + " highMask=" + Integer.toBinaryString(highMask));
    int hashPos = findHash(bytes, hashCode);
    //System.out.println("  hashPos=" + hashPos);
    int e = ids[hashPos];

    if (e == -1) {
      // new entry: append to off-heap, record ordinal, store high hash bits in ids
      long addr = pool.addBytesRef(bytes);
      ensureBytesStartCapacity(count + 1);
      bytesStart[count] = addr;

      e = count++;
      ids[hashPos] = e | (hashCode & highMask);

      if (count == hashHalfSize) {
        // nocommit to truly be able to max out hash we need to not double at the very end?
        //   or we paginate the hash arrays
        rehash(hashSize << 1, true);
      }
      return e;
    } else {
      // duplicate
      e = e & hashMask;
      return -(e + 1);
    }
  }

  /** Return the ordinal for the given BytesRef or -1 if not present. */
  public int find(BytesRef bytes) {
    int hashCode = doHash(bytes);
    //System.out.println("hashCode=" + hashCode);
    int id = ids[findHash(bytes, hashCode)];
    return id == -1 ? -1 : id & hashMask;
  }

  /**
   * Copy bytes for the given ordinal into {@code out}. {@code out.bytes} will be grown and {@code
   * out.offset} set to 0.
   */
  public void get(int ord, BytesRef out) {
    checkOrd(ord);
    pool.get(bytesStart[ord], out);
  }

  /**
   * Returns the ordinals currently stored in the table (in hash-table scan order). Entries are the
   * true ords (0..count-1), not including the cached hash bits.
   */
  public int[] compact() {
    int[] result = new int[count];
    int upto = 0;
    for (int i = 0; i < hashSize; i++) {
      final int v = ids[i];
      if (v != -1) {
        result[upto++] = v & hashMask;
      }
    }
    if (upto != count) {
      // shouldn't happen, but be safe
      result = Arrays.copyOf(result, upto);
    }
    return result;
  }

  /**
   * Memory used by on-heap structures (arrays); off-heap mapped capacity is not counted as heap.
   */
  @Override
  public long ramBytesUsed() {
    long bytes = 0;
    bytes += RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
    bytes += RamUsageEstimator.sizeOf(ids);
    bytes += RamUsageEstimator.sizeOf(bytesStart);
    // Note: off-heap memory-mapped file isn't heap; but expose its cap via pool.ramBytesUsedOffHeap
    // if needed.
    return bytes;
  }

  /** Bytes actually written into the off-heap file (append pointer). */
  public long offHeapBytesUsed() {
    return pool.used();
  }

  /** Best-effort accounting of mapped file capacity (off-heap), in bytes. */
  public long offHeapBytesCapacity() {
    return pool.capacity();
  }

  @Override
  public void close() throws IOException {
    pool.close();
  }

  // ---------------- internal methods ----------------

  private void ensureBytesStartCapacity(int min) {
    if (min > bytesStart.length) {
      bytesStart = ArrayUtil.grow(bytesStart, min);
    }
  }

  private static int doHash(BytesRef bytes) {
    return StringHelper.murmurhash3_x86_32(bytes.bytes, bytes.offset, bytes.length, StringHelper.GOOD_FAST_HASH_SEED);
  }

  /**
   * Probe to find slot for given BytesRef/hashCode; uses cached high bits to skip mismatches fast.
   */
  private int findHash(BytesRef bytes, int hashCode) {
    assert hashCode == doHash(bytes);
    int code = hashCode;
    int pos = code & hashMask;
    int e = ids[pos];
    int highBits = hashCode & highMask;
    if (e != -1) {
      //System.out.println("bytesStart=" + bytesStart[e & hashMask]);
      //System.out.println("  initial pos=" + pos + " e=" + e + " highBits=" + highBits + " higheq=" + ((e & highMask) == highBits) + " pooleq=" + pool.equals(bytesStart[e & hashMask], bytes));
    }
    while (e != -1 && ((e & highMask) != highBits || pool.equals(bytesStart[e & hashMask], bytes) == false)) {
      code++;
      pos = code & hashMask;
      e = ids[pos];
    }
    return pos;
  }

  private void rehash(final int newSize, boolean hashOnData) {
    final int newMask = newSize - 1;
    final int newHighMask = ~newMask;
    final int[] newHash = new int[newSize];
    Arrays.fill(newHash, -1);


    System.out.println("rehash to " + newSize);
    for (int i = 0; i < hashSize; i++) {
      int e0 = ids[i];
      if (e0 != -1) {
        e0 &= hashMask; // strip cached hash bits to recover the ordinal

        final int hashCode;
        int code;
        if (hashOnData) {
          hashCode = code = pool.hash(bytesStart[e0]);
        } else {
          hashCode = 0;
          code = (int) (bytesStart[e0] ^ (bytesStart[e0] >>> 32)); // fallback, not used in practice
        }

        int pos = code & newMask;
        while (newHash[pos] != -1) {
          code++;
          pos = code & newMask;
        }
        newHash[pos] = e0 | (hashCode & newHighMask);
      }
    }

    this.ids = newHash;
    this.hashSize = newSize;
    this.hashHalfSize = newSize >>> 1;
    this.hashMask = newMask;
    this.highMask = newHighMask;
  }

  private void checkOrd(int ord) {
    if (ord < 0 || ord >= count) {
      throw new ArrayIndexOutOfBoundsException("ord=" + ord + " not in [0.." + (count - 1) + "]");
    }
  }

  // ---------------- off-heap pool ----------------

  private static final class OffHeapFile implements Closeable {
    private final Path path;
    private final FileChannel ch;

    private Arena arena; // mapping arena
    private MemorySegment mapped; // full-file mapping
    private long capacity; // mapped/allocated bytes
    private long writePos; // append pointer

    OffHeapFile(Path dir) throws IOException {
      this.path =
          Files.createTempFile(
              dir == null ? Path.of(System.getProperty("java.io.tmpdir")) : dir,
              "offheap-bytesrefhash",
              ".bin");
      //System.out.println("temp path=" + this.path);
      this.ch = FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.DELETE_ON_CLOSE);
      /*
      this.ch =
        new RandomAccessFile(path.toString(), "rw").getChannel();
      */
      //System.out.println("opened " + this.ch);
      mapToCapacity(Math.max(1 << 20, 4096)); // start with 1 MB
    }

    long capacity() {
      return capacity;
    }

    long used() {
      return writePos;
    }

    void clear() {
      writePos = 0L;
      // data remains; logically cleared
      // TODO: should we tell OS to truncate the file, and lower capacity?
    }

    long addBytesRef(BytesRef br) {

      long addr = writePos;

      // header -- 1 or 2 byte vInt
      writeLenHeader(br.length);

      // payload
      // write in simple loop (fast enough; can be optimized in bulk with asByteBuffer if desired)
      /*
      final byte[] b = br.bytes;
      final int off = br.offset;
      for (int i = 0; i < br.length; i++) {
        mapped.set(ValueLayout.JAVA_BYTE, p + i, b[off + i]);
      }
      */

      ensureCapacity(writePos + br.length);
      
      MemorySegment.copy(br.bytes, br.offset, mapped, ValueLayout.JAVA_BYTE, writePos, br.length);

      writePos += br.length;
      return addr;
    }

    boolean equals(long addr, BytesRef candidate) {
      LenHeader header = readLenHeader(mapped, addr);
      int len = header.length;
      //System.out.println("  equals len=" + header.length + " nb=" + header.numBytes);
      if (len != candidate.length) return false;
      long p = addr + header.numBytes;
      byte[] b = candidate.bytes;
      int off = candidate.offset;
      for (int i = 0; i < len; i++) {
        byte v = mapped.get(ValueLayout.JAVA_BYTE, p + i);
        if (v != b[off + i]) return false;
      }
      return true;
    }

    void get(long addr, BytesRef out) {
      LenHeader header = readLenHeader(mapped, addr);
      int len = header.length;
      long p = addr + header.numBytes;
      if (out.bytes == null || out.bytes.length < len) {
        out.bytes = new byte[ArrayUtil.oversize(len, 1)];
      }
      out.offset = 0;
      out.length = len;
      for (int i = 0; i < len; i++) {
        out.bytes[i] = mapped.get(ValueLayout.JAVA_BYTE, p + i);
      }
    }

    int hash(long addr) {
      LenHeader header = readLenHeader(mapped, addr);
      int len = header.length;
      long p = addr + header.numBytes;
      // copy into temp array for StringHelper.murmurhash3_x86_32
      byte[] tmp = new byte[len];
      for (int i = 0; i < len; i++) {
        tmp[i] = mapped.get(ValueLayout.JAVA_BYTE, p + i);
      }
      return StringHelper.murmurhash3_x86_32(tmp, 0, len, StringHelper.GOOD_FAST_HASH_SEED);
    }

    private void ensureCapacity(long min) throws OutOfMemoryError {
      if (min <= capacity) return;
      long newCap = growCapacity(capacity, min);
      try {
        mapToCapacity(newCap);
      } catch (IOException ioe) {
        throw new OutOfMemoryError("failed to grow mapped file to " + newCap + " bytes: " + ioe);
      }
    }

    private void mapToCapacity(long newCap) throws IOException {
      // extend file
      System.out.println("grow map to " + newCap);
      if (ch.size() < newCap) {
        // reliably grow: position and write one byte -- how nice it has to have a file system abstraction to manage this storage!
        //System.out.println("extend " + ch.size() + " to " + (newCap-1));
        ch.position(newCap - 1);
        ch.write(ByteBuffer.wrap(new byte[] {0}));
        //System.out.println("size now " + ch.size());
      }
      // remap
      if (arena != null) {
        //System.out.println("close old arena");
        arena.close(); // unmap old segment
      }
      arena = Arena.ofConfined();
      //System.out.println("map capacity=" + capacity + " newCap=" + newCap);
      mapped = ch.map(FileChannel.MapMode.READ_WRITE, 0L, newCap, arena);
      capacity = newCap;
    }

    @Override
    public void close() throws IOException {
      if (arena != null) {
        try {
          // flush mapped contents
          if (mapped != null && mapped.isMapped()) {
            mapped.force();
          }
        } finally {
          arena.close(); // unmap
          arena = null;
        }
      }
      ch.close();

      // let IOException propogate
      Files.deleteIfExists(path);
    }

    private static long growCapacity(long cur, long min) {
      if (cur == 0) cur = 4096;
      long newCap = cur + (cur >>> 1); // ~1.5x like ArrayUtil.grow
      if (newCap < min) newCap = min;
      // guard overflow
      if (newCap < 0) newCap = Long.MAX_VALUE;
      return newCap;
    }

    // ---------------- length header (1 or 2 byte "vint") ----------------

    private void writeLenHeader(int len) {
      
      assert len <= MAX_VALUE_LENGTH;

      if (len < 0x80) {
        ensureCapacity(writePos + 1);
        mapped.set(ValueLayout.JAVA_BYTE, writePos, (byte) len);
        writePos += 1;
      } else {
        // two bytes: MSB of first indicates "more", remaining 7 bits are high bits of length, then
        // low 8 bits
        ensureCapacity(writePos + 2);
        mapped.set(ValueLayout.JAVA_BYTE, writePos, (byte) (0x80 | (len >>> 8)));
        mapped.set(ValueLayout.JAVA_BYTE, writePos + 1, (byte) (len & 0xFF));
        writePos += 2;
      }
    }

    private record LenHeader(int length, int numBytes) {};

    private LenHeader readLenHeader(MemorySegment seg, long pos) {
      int b0 = seg.get(ValueLayout.JAVA_BYTE, pos) & 0xFF;
      if ((b0 & 0x80) == 0) {
        return new LenHeader(b0, 1);
      } else {
        int b1 = seg.get(ValueLayout.JAVA_BYTE, pos + 1) & 0xFF;
        int len = ((b0 & 0x7F) << 8) | b1;
        return new LenHeader(len, 2);
      }
    }
  }

  public static class MaxBytesLengthExceededException extends IllegalArgumentException {
    public MaxBytesLengthExceededException(String msg) {
      super(msg);
    }
  }
}
