const API = "http://localhost:8001";

export interface ScoredRepo {
  id: number;
  full_name: string;
  description: string | null;
  language: string | null;
  topics: string[];
  stars: number;
  score: number;
  features: Record<string, number>;
  shared_topics: string[];
  shared_language: boolean;
}

export interface RecommendResponse {
  source_repo: string;
  repos: ScoredRepo[];
}

export async function fetchRecommendations(
  repo: string,
  limit = 10,
  seed?: number,
): Promise<RecommendResponse> {
  const params = new URLSearchParams({ repo, limit: String(limit) });
  if (seed !== undefined) params.set("seed", String(seed));
  const res = await fetch(`${API}/recommend?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      (err as { detail?: string }).detail || `API error ${res.status}`,
    );
  }
  return res.json();
}

export async function fetchExplore(
  seed: number,
  limit = 10,
): Promise<RecommendResponse> {
  const params = new URLSearchParams({
    seed: String(seed),
    limit: String(limit),
  });
  const res = await fetch(`${API}/explore?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      (err as { detail?: string }).detail || `API error ${res.status}`,
    );
  }
  return res.json();
}
