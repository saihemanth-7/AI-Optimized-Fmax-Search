/**
 * analytics.js
 * Controls image modal zoom and analytics interactions on the Analytics & Graphs page.
 */

let modalInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    const modalEl = document.getElementById('imageModal');
    if (modalEl && window.bootstrap) {
        modalInstance = new bootstrap.Modal(modalEl);
    }
});

function openImageModal(imgSrc, title) {
    const modalImg = document.getElementById('imageModalImg');
    const modalTitle = document.getElementById('imageModalTitle');

    if (modalImg && modalTitle) {
        modalImg.src = imgSrc;
        modalTitle.textContent = title;
        if (modalInstance) {
            modalInstance.show();
        }
    }
}

