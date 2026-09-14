lucide.createIcons();

// File input label updater
const fileInput = document.getElementById('video_file');
if (fileInput) {
  fileInput.addEventListener('change', () => {
    document.getElementById('file-label').innerText =
      fileInput.files[0]?.name || 'Choose a file to analyze';
  });
}

// Live clock (HH:MM:SS)
function updateClock() {
  const now = new Date();
  document.getElementById('clock').innerText = now.toTimeString().split(' ')[0];
}
setInterval(updateClock, 1000);
updateClock();

// Incident log polling (every 2.5 seconds)
async function fetchLogs() {
  try {
    const response = await fetch('/api/events');
    const data = await response.json();
    const tableBody = document.getElementById('log-table');

    if (!data || data.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="6" class="log-empty">No incidents recorded. All clear.</td></tr>`;
      return;
    }

    tableBody.innerHTML = '';
    data.forEach(row => {
      const angleText = (row.torso_angle !== null && row.torso_angle !== undefined)
        ? `${parseFloat(row.torso_angle).toFixed(1)}°`
        : 'N/A';

      const arText = (row.aspect_ratio !== null && row.aspect_ratio !== undefined)
        ? parseFloat(row.aspect_ratio).toFixed(2)
        : 'N/A';

      tableBody.innerHTML += `
        <tr>
          <td class="mono">#${row.id.toString().padStart(4, '0')}</td>
          <td class="mono">${row.time}</td>
          <td class="mono" style="color: var(--danger); font-weight: 500;">${angleText}</td>
          <td class="mono">${arText}</td>
          <td style="color: var(--danger);">Fall confirmed</td>
          <td class="align-right mono" style="color: var(--safe); font-size: 12px;">${row.status}</td>
        </tr>`;
    });
  } catch (err) {
    console.error("Error polling /api/events:", err);
  }
}

// Initial call and periodic polling
fetchLogs();
setInterval(fetchLogs, 2500);