const QUIZ_ITEMS = [
  {
    headline: "Officials confirmed the new transit funding bill will move forward after a committee review, with a formal vote expected within three weeks.",
    answer: "REAL",
    explain: "Measured tone, attribution to officials, and a specific procedural timeline are typical of sourced reporting."
  },
  {
    headline: "SHOCKING: Doctors don't want you to know this ONE secret that the government has hidden for 50 years — share before it's DELETED!",
    answer: "FAKE",
    explain: "Urgency ('before it's deleted'), vague authority ('the government'), and all-caps emphasis are classic clickbait/misinformation markers."
  },
  {
    headline: "A report from the Bureau of Labor Statistics shows unemployment shifted by 0.3% compared to last quarter, based on data from 400 regional offices.",
    answer: "REAL",
    explain: "Specific, checkable numbers and a named data source are hallmarks of real statistical reporting."
  },
  {
    headline: "BREAKING: Anonymous insider reveals secret plan that changes EVERYTHING — mainstream media REFUSES to cover this!",
    answer: "FAKE",
    explain: "Anonymous unverifiable sourcing plus a claim that 'media refuses to cover it' is a common misinformation framing device."
  },
  {
    headline: "Lawmakers debated the proposed energy bill for several hours before agreeing to send it to committee for further review next month.",
    answer: "REAL",
    explain: "Describes an ordinary legislative process without dramatic claims — consistent with routine real reporting."
  },
  {
    headline: "You won't BELIEVE what scientists just discovered — this simple trick makes doctors furious, and they don't want it to spread!",
    answer: "FAKE",
    explain: "'You won't believe', vague 'scientists', and 'doctors are furious' are emotionally manipulative, non-specific phrasing typical of fabricated content."
  }
];

let score = 0;
let answered = 0;

function renderQuiz() {
  const container = document.getElementById("quizContainer");
  container.innerHTML = "";

  QUIZ_ITEMS.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "quiz-card";
    card.innerHTML = `
      <p class="quiz-headline">"${item.headline}"</p>
      <div class="quiz-options">
        <button class="quiz-btn" data-choice="REAL" data-idx="${idx}">REAL</button>
        <button class="quiz-btn" data-choice="FAKE" data-idx="${idx}">FAKE</button>
      </div>
      <p class="quiz-explain" id="explain-${idx}">${item.explain}</p>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".quiz-btn").forEach(btn => {
    btn.addEventListener("click", handleAnswer);
  });
}

function handleAnswer(e) {
  const idx = parseInt(e.target.dataset.idx);
  const choice = e.target.dataset.choice;
  const item = QUIZ_ITEMS[idx];
  const card = e.target.closest(".quiz-card");
  const buttons = card.querySelectorAll(".quiz-btn");

  if (card.dataset.answered) return; // already answered
  card.dataset.answered = "true";

  buttons.forEach(b => {
    b.disabled = true;
    if (b.dataset.choice === item.answer) b.classList.add("correct");
    else if (b.dataset.choice === choice) b.classList.add("incorrect");
  });

  document.getElementById(`explain-${idx}`).style.display = "block";

  answered++;
  if (choice === item.answer) score++;

  if (answered === QUIZ_ITEMS.length) {
    const scoreBox = document.getElementById("quizScore");
    scoreBox.style.display = "block";
    scoreBox.textContent = `You scored ${score} / ${QUIZ_ITEMS.length}`;
  }
}

renderQuiz();
