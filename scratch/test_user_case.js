
const rawText = `CASE ASSESSMENT:
...
FIELD NOTE:
This case requires further investigation to determine the true nature of the observed phenomenon and to rule out other explanations.

FOLLOW_UP_QUESTIONS:
["What does immaterial mean in this context?", "Why was environmental probability so low?", "Could this be a misclassification?"]`;

const markerRegex = /\n?\s*(?:\*\*)?FOLLOW[-_ ]?UP[-_ ]?QUESTIONS\s*:(?:\*\*)?\s*/i;
const match = rawText.match(markerRegex);

if (match) {
    const idx = match.index;
    const cleanText = rawText.slice(0, idx).trimEnd();
    const jsonPart = rawText.slice(idx + match[0].length).trim();
    console.log('Match found!');
    console.log('Clean Text End:', cleanText.slice(-50));
    console.log('JSON Part Start:', jsonPart.slice(0, 50));
} else {
    console.log('No match found');
}
