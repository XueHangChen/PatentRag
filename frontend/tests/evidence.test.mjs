import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(testDir, "..");

async function loadEvidenceModule() {
  const result = await build({
    bundle: true,
    entryPoints: [path.join(frontendRoot, "src", "evidence.ts")],
    format: "esm",
    platform: "node",
    write: false
  });
  const moduleText = result.outputFiles[0].text;
  const moduleUrl = `data:text/javascript;base64,${Buffer.from(moduleText).toString("base64")}`;

  return import(moduleUrl);
}

test("formatEvidenceScore renders finite scores with two decimals", async () => {
  const { formatEvidenceScore } = await loadEvidenceModule();

  assert.equal(formatEvidenceScore(0.756), "0.76");
  assert.equal(formatEvidenceScore(Number.NaN), "0.00");
});

test("buildTechnicalEvidenceGroups keeps only populated evidence groups", async () => {
  const { buildTechnicalEvidenceGroups } = await loadEvidenceModule();

  const groups = buildTechnicalEvidenceGroups({
    technical_fields: ["医疗器械"],
    problems: ["喷雾不均匀"],
    components: [],
    solutions: ["雾化喷嘴"],
    effects: []
  });

  assert.deepEqual(groups, [
    { label: "技术领域", values: ["医疗器械"] },
    { label: "技术问题", values: ["喷雾不均匀"] },
    { label: "技术方案", values: ["雾化喷嘴"] }
  ]);
});

test("buildTechnicalEvidenceGroups tolerates missing graph evidence arrays", async () => {
  const { buildTechnicalEvidenceGroups } = await loadEvidenceModule();

  const groups = buildTechnicalEvidenceGroups({
    technical_fields: ["样本运输"]
  });

  assert.deepEqual(groups, [{ label: "技术领域", values: ["样本运输"] }]);
});
