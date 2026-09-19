# Vdbench FWD skew denominator truncation

Observed on the stock jar with SHA-256
`8d53b728baf4e3eb28b538765b81de606ce2b1dfca39c66822d02704b462304a`.
The CloudLab fractional-transfer overlay does not change `SkewReport`.

## Binary evidence

`SkewReport.javap.txt`, method `reportFileEndOfRunSkew`:

- Bytecodes 0-1 initialize a long accumulator.
- Bytecodes 43-55 convert that long to double, add `getReqstdRate(): double`,
  then execute `d2l` and store back to the long for every FWD.
- Bytecodes 725-734 divide the original double per-FWD rate by that truncated
  accumulator to produce the displayed actual percent.

Equivalent behavior, reconstructed from bytecode:

```java
long totalRate = 0;
for (FwdStats stats : workloads) {
    totalRate += stats.getReqstdRate();
}
double actualPercent = 100 * stats.getReqstdRate() / totalRate;
```

With many FWDs below one operation per second, most contributions disappear from
the denominator. This explains actual-percent sums above 100%, not an actual
workload capable of producing 144% of its operations.

## Independent correction

`audit_vdbench.py` sums integer bucket counts in every per-FWD histogram and
checks their sum against the overall histogram for each RD. It then calculates
shares and differences from the exact model weights. The method is validated
against the supplied historical reports as well as CloudLab reports.

No live binary or original report was modified after the experiment began.
The proper upstream repair is a double accumulator or, preferably, integer
operation-count aggregation with one final division. A reporting-only repair
does not guarantee that the realized workload matches its requested shares.

The original colleague's missing Vdbench binary cannot be disassembled here;
its supplied skew reports show the same inconsistency, but byte-identical
implementation is not claimed.
