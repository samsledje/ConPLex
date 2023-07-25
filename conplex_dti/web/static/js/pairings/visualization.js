// The caller must set the following global variables to URLs that return JSON:
// `drug_projections_url`
// `target_projections_url`
// `predictions_url`

// The caller must have the following elements:
// a `span` element with ID `selected-drug`
// a `span` element with ID `selected-target`
// a `ul` element with ID `drugs`
// a `ul` element with ID `targets`
// a `span` element with ID `status`
// a `span` element with ID `prediction`
// a relatively positioned `div` element with ID `visualization`

const statusElement = document.getElementById("status")
function setStatus(status) {
    statusElement.innerHTML = status
}
