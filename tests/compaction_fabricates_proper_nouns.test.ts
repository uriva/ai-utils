import { assertEquals } from "@std/assert";
import { sum } from "gamla";
import { z } from "zod/v4";
import type { HistoryEvent } from "../src/agent.ts";
import { eventsToPlainText, summarizeEvents } from "../src/compaction.ts";
import { geminiGenJson } from "../src/gemini.ts";
import { injectSecrets, llmTest } from "../test_helpers.ts";

// Repro target: the compactor sometimes takes history that only refers to an
// entity generically (e.g. "the second hotel near X") and crystallizes it into
// a fabricated proper-noun ("X-something Hotel") in the summary. Downstream
// turns then treat that fabricated name as established fact.
//
// First failure point per AGENTS.md: a single call to `summarizeEvents`
// produces a summary containing specific proper-noun entities (hotel names,
// person names, document names) that are NOT present anywhere in the source
// events.

const makeOwn = (text: string, timestamp: number): HistoryEvent => ({
  id: crypto.randomUUID(),
  type: "own_utterance",
  text,
  timestamp,
  isOwn: true,
});

const makeUser = (text: string, timestamp: number): HistoryEvent => ({
  id: crypto.randomUUID(),
  type: "participant_utterance",
  text,
  timestamp,
  isOwn: false,
  name: "traveler",
});

const buildEvents = (neighborhood: string): HistoryEvent[] => {
  const base = Date.now();
  const t = (n: number) => base + n * 1000;
  return [
    makeUser(
      `I'm planning a Tokyo trip with my friend. We need two hotels: one downtown for the first part, and a second one near ${neighborhood} for later. I'm a professional photographer and want stylish places, not student vibes.`,
      t(0),
    ),
    makeOwn(
      `Got it — boutique/design-leaning rather than backpacker, both downtown and near ${neighborhood}. I'll keep your photography work in mind too (good light, walkable to interesting streetscapes).`,
      t(1),
    ),
    makeUser(
      `My friend already booked her own hotel. She sent me the details in a message.`,
      t(2),
    ),
    makeOwn(
      `OK — so we only need to coordinate yours, and we can pick somewhere near her base if you want to meet up easily. Where is she staying?`,
      t(3),
    ),
    makeUser(
      `She's near the convention center for work. I'll forward the message later.`,
      t(4),
    ),
    makeOwn(
      `Sounds good. For now let's focus on your two stays: downtown first, then near ${neighborhood}.`,
      t(5),
    ),
    makeUser(`How many nights does it make sense for each?`, t(6)),
    makeOwn(
      `Plan was 3 nights at the first hotel (downtown), then 4 nights at the second hotel (${neighborhood} area). We can adjust.`,
      t(7),
    ),
    makeUser(`What about a day trip out of Tokyo somewhere coastal?`, t(8)),
    makeOwn(
      `There are several great coastal day trips reachable by train in 1-2 hours — temples, beaches, old castle towns. We can pick one once the hotels are locked.`,
      t(9),
    ),
    makeUser(`I might also want a mountain shrine day.`, t(10)),
    makeOwn(
      `Doable — there's a famous mountain shrine area 2 hours north with cedar forests, very photogenic. Good for a sunrise visit.`,
      t(11),
    ),
    makeUser(
      `OK back to hotels. The first one — quiet downtown is fine, but I want good morning light for window shots.`,
      t(12),
    ),
    makeOwn(
      `Noted — for the downtown stay we'll prioritize east-facing higher floors. For the second hotel near ${neighborhood} we can pick something with a rooftop or a quiet side street view.`,
      t(13),
    ),
    makeUser(`כמה זה עולה בערך ללילה?`, t(14)),
    makeOwn(
      `בוטיק/עיצובי באזורים האלה זה בערך 25,000-45,000 ין ללילה לחדר זוגי. את המלון הראשון (downtown) אפשר לפעמים למצוא קצת יותר זול, את השני (${neighborhood}) דווקא יותר מבוקש.`,
      t(15),
    ),
    makeUser(`And what's the closest station to the second hotel area?`, t(16)),
    makeOwn(
      `The ${neighborhood} area is served by the JR Yamanote line plus a couple of subway lines, so transit from most museum districts is 15-25 minutes.`,
      t(17),
    ),
    makeUser(
      `My friend texted me the name of her hotel but it's in Russian, I'll paste it later.`,
      t(18),
    ),
    makeOwn(
      `No problem, I can translate it. Just paste when you have it.`,
      t(19),
    ),
    makeUser(`Also I want to do at least one big art museum.`, t(20)),
    makeOwn(
      `Tokyo has several major museums — modern art, photography, classical. I'd recommend at least the photography museum given your work, and one modern art museum.`,
      t(21),
    ),
    makeUser(`Photography museum yes definitely.`, t(22)),
    makeOwn(
      `Locked in. We'll slot it in on a day with afternoon free time near the second hotel.`,
      t(23),
    ),
    makeUser(
      `Let's wrap up the first hotel decision soon, I want to book this week.`,
      t(24),
    ),
    makeOwn(
      `Understood — I'll come back to you with 2-3 concrete options for the first hotel (downtown) and 2-3 for the second hotel (${neighborhood} area) so you can pick.`,
      t(25),
    ),
    makeUser(`Perfect.`, t(26)),
  ];
};

const judgeSchema = z.object({
  fabricatedProperNouns: z.array(z.string()).describe(
    "List of specific proper-noun entities (hotel names, person names, document/text titles, restaurant names, brand-name landmarks) that appear in the SUMMARY but do NOT appear in the SOURCE EVENTS. Do not include generic descriptions like 'the second hotel' or neighborhood names that ARE in the source. Only include genuinely fabricated specific names. If none, return empty array.",
  ),
  reasoning: z.string().describe(
    "One short sentence explaining your decision.",
  ),
});

const judgeFabrications = (source: string, summary: string) =>
  geminiGenJson(
    { mini: true },
    `You are checking whether a conversation summary fabricated specific named entities (proper nouns) that were not present in the original conversation events.

You will be given:
1. SOURCE EVENTS: the raw conversation text.
2. SUMMARY: a structured summary produced from those events.

Your job: list every PROPER-NOUN entity that the SUMMARY introduces but that does NOT appear in the SOURCE EVENTS. Specifically watch for: hotel names, person names, document/text titles, specific restaurant or shop names, named tours or services. IGNORE generic phrases ("the second hotel", "the downtown hotel"), neighborhood/city names that ARE in the source, and common nouns. Only flag fabricated SPECIFIC proper-noun entities.`,
    judgeSchema,
  )(`SOURCE EVENTS:\n${source}\n\nSUMMARY:\n${summary}`);

const probe = async (neighborhood: string) => {
  const events = buildEvents(neighborhood);
  const source = eventsToPlainText(events);
  const summary = await summarizeEvents(events);
  const verdict = await judgeFabrications(source, summary);
  return { neighborhood, summary, verdict };
};

const neighborhoods = ["Ebisu", "Shibuya", "Asakusa", "Ginza", "Ueno"];

llmTest(
  "summarizeEvents does not fabricate proper-noun entities absent from source events",
  injectSecrets(async () => {
    const probes = await Promise.all(neighborhoods.map(probe));
    probes.forEach((p) =>
      console.log(
        `[${p.neighborhood}] fabricated=${
          JSON.stringify(p.verdict.fabricatedProperNouns)
        } reason=${p.verdict.reasoning}`,
      )
    );
    const totalFabricated = sum(
      probes.map((p) => p.verdict.fabricatedProperNouns.length),
    );
    const offenders = probes.filter((p) =>
      p.verdict.fabricatedProperNouns.length > 0
    );
    assertEquals(
      totalFabricated,
      0,
      `Compactor fabricated proper-noun entities in ${offenders.length}/${probes.length} probes:\n${
        offenders.map((o) =>
          `  - ${o.neighborhood}: ${
            JSON.stringify(o.verdict.fabricatedProperNouns)
          }\n      reason: ${o.verdict.reasoning}\n      summary excerpt: ${
            o.summary.slice(0, 600).replace(/\n/g, " ")
          }`
        ).join("\n")
      }`,
    );
  }),
  1,
);
