/*!
 * Toggle the navtree sync button twice during page load.
 * This resets Doxygen's internal sync state so the
 * navigation pane stays synchronized and the icon shows
 * the correct status.
 */
window.addEventListener('load', function() {
    var navSync = document.getElementById('nav-sync');
    if (!navSync) return;
    try {
        navSync.click();
    } catch (e) {
        /* ignore */
    }
});