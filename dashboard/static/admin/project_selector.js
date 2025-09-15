document.addEventListener("DOMContentLoaded", function () {
    const selector = document.getElementById("id_project_selector");
    const baserowInput = document.getElementById("id_baserow_id");
    const nameInput = document.getElementById("id_name");
    const logoInput = document.getElementById("id_logo_url");

    // This variable will be injected by Django in the template
    const projectDataEl = document.getElementById("project-data-json");
    if (!projectDataEl) return;
    const projects = JSON.parse(projectDataEl.textContent);

    selector.addEventListener("change", function () {
        const selectedId = this.value;
        if (!selectedId) {
            baserowInput.value = "";
            nameInput.value = "";
            logoInput.value = "";
            return;
        }
        const project = projects.find(p => String(p.id) === String(selectedId));
        if (project) {
            baserowInput.value = project.id;
            nameInput.value = project.name;
            logoInput.value = project.logo || "";
        }
    });
});
