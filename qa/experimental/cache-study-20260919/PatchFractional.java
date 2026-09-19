import java.nio.file.Files;
import java.nio.file.Path;
import java.util.jar.JarFile;
import jdk.internal.org.objectweb.asm.*;

// Retain every original binary field and method except the faulty sampler body.
public final class PatchFractional implements Opcodes {
  public static void main(String[] args) throws Exception {
    byte[] original;
    try (JarFile jar = new JarFile(args[0])) {
      original = jar.getInputStream(jar.getJarEntry("Vdb/FwgEntry.class")).readAllBytes();
    }
    ClassReader reader = new ClassReader(original);
    ClassWriter writer = new ClassWriter(reader, ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
    int[] changed = {0};
    reader.accept(new ClassVisitor(ASM7, writer) {
      @Override public MethodVisitor visitMethod(int access, String name, String desc,
                                                 String signature, String[] exceptions) {
        MethodVisitor m = super.visitMethod(access, name, desc, signature, exceptions);
        if (!name.equals("getXferSize") || !desc.equals("()I")) return m;
        changed[0]++;
        m.visitCode();
        m.visitMethodInsn(INVOKESTATIC, "Vdb/Validate", "isJournalRecoveryActive", "()Z", false);
        Label normal = new Label();
        m.visitJumpInsn(IFEQ, normal);
        m.visitVarInsn(ALOAD, 0);
        m.visitFieldInsn(GETFIELD, "Vdb/FwgEntry", "anchor", "LVdb/FileAnchor;");
        m.visitMethodInsn(INVOKEVIRTUAL, "Vdb/FileAnchor", "getDVMap", "()LVdb/DV_map;", false);
        m.visitMethodInsn(INVOKEVIRTUAL, "Vdb/DV_map", "getKeyBlockSize", "()I", false);
        m.visitInsn(IRETURN);
        m.visitLabel(normal);
        m.visitVarInsn(ALOAD, 0);
        m.visitFieldInsn(GETFIELD, "Vdb/FwgEntry", "xfer_size_randomizer", "Ljava/util/Random;");
        m.visitVarInsn(ALOAD, 0);
        m.visitFieldInsn(GETFIELD, "Vdb/FwgEntry", "xfersizes", "[D");
        m.visitMethodInsn(INVOKESTATIC, "Vdb/FractionalSampler", "sample", "(Ljava/util/Random;[D)I", false);
        m.visitInsn(IRETURN);
        m.visitMaxs(0, 0);
        m.visitEnd();
        return null;
      }
    }, 0);
    if (changed[0] != 1) throw new AssertionError("expected exactly one sampler");
    Path output = Path.of(args[1], "Vdb", "FwgEntry.class");
    Files.createDirectories(output.getParent());
    Files.write(output, writer.toByteArray());
  }
}
