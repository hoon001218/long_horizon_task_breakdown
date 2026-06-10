const form = document.getElementById("plannerForm");
const imageInput = document.getElementById("imageInput");
const attachButton = document.getElementById("attachButton");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const imagePreview = document.getElementById("imagePreview");
// const previewCaption = document.getElementById("previewCaption");
const actionList = document.getElementById("actionList");
const userPrompt = document.getElementById("userPrompt");
const runButton = document.getElementById("runButton");

let selectedFile = null;

const PREVIEW_MAX_WIDTH = 1200;
const PREVIEW_MAX_HEIGHT = 720;

function renderPlaceholder(message) {
  actionList.innerHTML = `
    <li class="action-card placeholder">
      <strong>${message.title}</strong>
      <p>${message.details}</p>
    </li>
  `;
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function resizeImageForPreview(file) {
  const imageUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(String(event.target?.result || ""));
    reader.onerror = () => reject(new Error("이미지 파일을 읽을 수 없습니다."));
    reader.readAsDataURL(file);
  });

  const image = await new Promise((resolve, reject) => {
    const loadedImage = new Image();
    loadedImage.onload = () => resolve(loadedImage);
    loadedImage.onerror = () => reject(new Error("이미지 미리보기를 생성할 수 없습니다."));
    loadedImage.src = imageUrl;
  });

  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const scale = Math.min(1, PREVIEW_MAX_WIDTH / sourceWidth, PREVIEW_MAX_HEIGHT / sourceHeight);
  const targetWidth = Math.max(1, Math.round(sourceWidth * scale));
  const targetHeight = Math.max(1, Math.round(sourceHeight * scale));

  const canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("미리보기 렌더링을 준비할 수 없습니다.");
  }

  context.drawImage(image, 0, 0, targetWidth, targetHeight);
  return canvas.toDataURL(file.type || "image/png", 0.92);
}

function renderActions(data) {
  const actions = Array.isArray(data.partial_actions) ? data.partial_actions : [];

  if (!actions.length) {
    renderPlaceholder({
      title: "결과 없음",
      details: "모델 응답에서 파싱 가능한 action list를 찾지 못했습니다.",
    });
    return;
  }

  actionList.innerHTML = actions
    .map((action, index) => {
      const title = escapeHtml(String(action.title || `Action ${index + 1}`));
      const details = escapeHtml(String(action.details || ""));
      return `
        <li class="action-card">
          <strong>${String(index + 1).padStart(2, "0")}. ${title}</strong>
          <p>${details || "세부 설명이 없습니다."}</p>
        </li>
      `;
    })
    .join("");
}

async function setPreview(file) {
  if (!file) {
    imagePreview.src = "";
    imagePreview.alt = "업로드된 이미지 미리보기";
    fileName.textContent = "선택된 파일 없음";
    return;
  }

  fileName.textContent = file.name;

  try {
    const resizedPreview = await resizeImageForPreview(file);
    imagePreview.src = resizedPreview;
    imagePreview.alt = file.name;
  } catch (error) {
    imagePreview.src = "";
    imagePreview.alt = "업로드된 이미지 미리보기";
    renderPlaceholder({
      title: "미리보기 오류",
      details: error instanceof Error ? error.message : "이미지를 표시할 수 없습니다.",
    });
  }
}

function setSelectedFile(file) {
  selectedFile = file;
  if (file) {
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    imageInput.files = dataTransfer.files;
  }
  setPreview(file);
}

attachButton.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0] || null;
  setSelectedFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("is-dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0] || null;
  if (file) {
    setSelectedFile(file);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!selectedFile) {
    renderPlaceholder({
      title: "이미지를 먼저 선택하세요",
      details: "File Attach 또는 Drag-Drop으로 이미지 파일을 넣어야 합니다.",
    });
    return;
  }

  const prompt = userPrompt.value.trim();
  if (!prompt) {
    renderPlaceholder({
      title: "프롬프트가 비어 있습니다",
      details: "유저 프롬프트를 입력한 뒤 다시 실행하세요.",
    });
    return;
  }

  const payload = new FormData();
  payload.append("image", selectedFile);
  payload.append("user_prompt", prompt);

  runButton.disabled = true;
  runButton.textContent = "실행 중...";
  renderPlaceholder({
    title: "요청 전송 중",
    details: "LLM 응답을 기다리는 중입니다.",
  });

  try {
    const response = await fetch("/api/partial-actions", {
      method: "POST",
      body: payload,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "서버 요청에 실패했습니다.");
    }

    renderActions(data);
  } catch (error) {
    actionList.innerHTML = `
      <li class="action-card error">
        <strong>오류</strong>
        <p>${escapeHtml(error instanceof Error ? error.message : String(error))}</p>
      </li>
    `;
  } finally {
    runButton.disabled = false;
    runButton.textContent = "실행";
  }
});

renderPlaceholder({
  title: "대기 중",
  details: "이미지와 프롬프트를 입력하고 실행 버튼을 누르면 결과가 여기에 쌓입니다.",
});