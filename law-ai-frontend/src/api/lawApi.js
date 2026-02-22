const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export async function askLawAI(question) {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error("Backend error");
  }

  return response.json();
}
