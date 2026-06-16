export type TechnicalEvidenceInput = {
  technical_fields?: string[];
  problems?: string[];
  components?: string[];
  solutions?: string[];
  effects?: string[];
};

export type EvidenceGroup = {
  label: string;
  values: string[];
};

export function formatEvidenceScore(score: number): string {
  return Number.isFinite(score) ? score.toFixed(2) : "0.00";
}

export function buildTechnicalEvidenceGroups(source: TechnicalEvidenceInput): EvidenceGroup[] {
  return [
    { label: "技术领域", values: source.technical_fields ?? [] },
    { label: "技术问题", values: source.problems ?? [] },
    { label: "关键组件", values: source.components ?? [] },
    { label: "技术方案", values: source.solutions ?? [] },
    { label: "技术效果", values: source.effects ?? [] }
  ].filter((group) => group.values.length > 0);
}
