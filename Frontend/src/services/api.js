const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/$/, "");


async function getErrorMessage(
  response,
  fallbackMessage,
) {
  try {
    const data = await response.json();

    if (typeof data.detail === "string") {
      return data.detail;
    }

    if (Array.isArray(data.detail)) {
      return data.detail
        .map((item) => {
          const location =
            item.loc?.join(".") ||
            "request";

          return `${location}: ${item.msg}`;
        })
        .join(", ");
    }

    return (
      data.message ||
      fallbackMessage
    );
  } catch {
    return fallbackMessage;
  }
}


export async function checkBackendHealth() {
  const response = await fetch(
    `${API_BASE_URL}/health`,
  );

  if (!response.ok) {
    throw new Error(
      "Backend is unavailable.",
    );
  }

  return response.json();
}


export async function sendMessage({
  question,
  userId,
  sessionId,
}) {
  const response = await fetch(
    `${API_BASE_URL}/api/chat`,
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        question,
        user_id: userId,
        session_id:
          sessionId || null,
      }),
    },
  );

  if (!response.ok) {
    const message =
      await getErrorMessage(
        response,
        "The assistant could not process your message.",
      );

    throw new Error(message);
  }

  return response.json();
}


export async function transcribeAudio({
  audioBlob,
  languageCode,
}) {
  if (
    !audioBlob ||
    audioBlob.size === 0
  ) {
    throw new Error(
      "No microphone recording was captured.",
    );
  }

  const formData =
    new FormData();

  const isOgg =
    audioBlob.type.includes("ogg");

  const extension =
    isOgg ? "ogg" : "webm";

  /*
   * This must match:
   *
   * audio: UploadFile = File(...)
   */
  formData.append(
    "audio",
    audioBlob,
    `voice-recording.${extension}`,
  );

  formData.append(
    "language_code",
    languageCode || "en-IN",
  );

  const response = await fetch(
    `${API_BASE_URL}/api/stt`,
    {
      method: "POST",
      body: formData,
    },
  );

  /*
   * Do not manually set Content-Type.
   * The browser creates the multipart boundary.
   */

  if (!response.ok) {
    const message =
      await getErrorMessage(
        response,
        "Speech transcription failed.",
      );

    throw new Error(message);
  }

  return response.json();
}


export async function synthesizeSpeech({
  text,
  languageCode,
}) {
  const cleanText = text?.trim();

  if (!cleanText) {
    throw new Error(
      "Text is required for speech generation.",
    );
  }

  const response = await fetch(
    `${API_BASE_URL}/api/tts`,
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        text: cleanText,
        language_code:
          languageCode || "en-IN",
      }),
    },
  );

  if (!response.ok) {
    const message =
      await getErrorMessage(
        response,
        "Voice generation failed.",
      );

    throw new Error(message);
  }

  return response.blob();
}