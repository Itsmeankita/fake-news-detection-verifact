chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "verifact-check",
    title: 'Check with VeriFact: "%s"',
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "verifact-check" && info.selectionText) {
    chrome.storage.local.set({ selectedText: info.selectionText }, () => {
      chrome.action.openPopup();
    });
  }
});
