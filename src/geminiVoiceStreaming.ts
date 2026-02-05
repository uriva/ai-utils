export type LiveSessionConfig = {
  apiKey: string;
  model?: string;
  systemInstruction?: string;
};

type LiveSessionState = {
  ws: WebSocket;
  model: string;
};

export const createLiveSession = async ({
  apiKey,
  model = "gemini-2.0-flash-exp",
  systemInstruction = "You are a helpful voice assistant. Keep responses concise and natural for voice conversation.",
}: LiveSessionConfig): Promise<LiveSessionState> => {
  const ws = new WebSocket(
    `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=${apiKey}`,
  );

  await new Promise<void>((resolve, reject) => {
    ws.onopen = () => {
      console.log("🔌 WebSocket connected");
      const setupMessage = {
        setup: {
          model: "models/gemini-2.5-flash-native-audio-preview-12-2025",
          generation_config: {
            response_modalities: ["AUDIO"],
            speech_config: {
              voice_config: {
                prebuilt_voice_config: {
                  voice_name: "Puck",
                },
              },
            },
          },
          system_instruction: {
            parts: [{ text: systemInstruction }],
          },
        },
      };
      console.log("📤 Sending setup:", JSON.stringify(setupMessage, null, 2));
      ws.send(JSON.stringify(setupMessage));
      resolve();
    };
    ws.onerror = (err) => {
      console.error("❌ WebSocket error:", err);
      reject(err);
    };
  });

  return {
    ws,
    model,
  };
};

export const sendAudioChunk = (
  state: LiveSessionState,
  audioBase64: string,
): void => {
  const message = {
    realtime_input: {
      media_chunks: [{
        mime_type: "audio/pcm",
        data: audioBase64,
      }],
    },
  };
  state.ws.send(JSON.stringify(message));
};

export const handleMessages = (
  state: LiveSessionState,
  onAudioResponse: (audioBase64: string) => void,
  onTextResponse: (text: string) => void,
  onTurnComplete: () => void,
  onError: (error: unknown) => void,
): void => {
  state.ws.onmessage = async (event) => {
    let data = event.data;
    
    if (data instanceof Blob) {
      data = await data.text();
    }
    
    const msg = JSON.parse(data);
    const hasContent = !!(msg.setupComplete || msg.serverContent);
    Deno.stdout.writeSync(new TextEncoder().encode(`📨 Message: ${hasContent ? 'has content' : 'empty'}\n`));

    if (msg.setupComplete) {
      Deno.stdout.writeSync(new TextEncoder().encode("✅ Setup complete\n"));
      return;
    }

    if (msg.serverContent?.modelTurn) {
      const parts = msg.serverContent.modelTurn.parts || [];
      Deno.stdout.writeSync(new TextEncoder().encode(`🎯 Model turn with ${parts.length} parts\n`));

      for (const part of parts) {
        if (part.inlineData?.data) {
          Deno.stdout.writeSync(new TextEncoder().encode("🔊 Received audio chunk\n"));
          onAudioResponse(part.inlineData.data);
        }
        if (part.text) {
          Deno.stdout.writeSync(new TextEncoder().encode(`💬 Received text: ${part.text}\n`));
          onTextResponse(part.text);
        }
      }
    }

    if (msg.serverContent?.turnComplete) {
      Deno.stdout.writeSync(new TextEncoder().encode("✅ Turn complete\n"));
      onTurnComplete();
    }

    if (msg.error) {
      Deno.stdout.writeSync(new TextEncoder().encode(`❌ Server error: ${JSON.stringify(msg.error)}\n`));
      onError(msg.error);
    }
  };

  state.ws.onerror = (event) => {
    Deno.stdout.writeSync(new TextEncoder().encode(`❌ WebSocket error event: ${JSON.stringify(event)}\n`));
    onError(event);
  };

  state.ws.onclose = (event) => {
    Deno.stdout.writeSync(new TextEncoder().encode(`🔌 WebSocket closed: ${event.code} ${event.reason}\n`));
  };
};

export const closeSession = (state: LiveSessionState): void => {
  state.ws.close();
};
