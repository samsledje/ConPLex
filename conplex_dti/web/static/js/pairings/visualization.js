// The caller must set the following global variables to URLs that return JSON:
// `drugProjectionsUrl`
// `targetProjectionsUrl`
// `predictionsUrl`

// The caller must have the following elements:
// a `span` element with ID `selected-drug`
// a `span` element with ID `selected-target`
// a `ul` element with ID `drugs`
// a `ul` element with ID `targets`
// a `span` element with ID `status`
// a `span` element with ID `prediction`
// a relatively positioned `div` element with ID `visualization`

const visualizationElement = document.getElementById("visualization")
const visualizationPadding = 0.1

const elementIdToDefault = {
    "selected-drug": "No drug selected.",
    "selected-target": "No target selected.",
    "drugs": "<li>No drugs loaded.</li>",
    "targets": "<li>No targets loaded.</li>",
    "status": "Building visualization.",
    "prediction": "No drug or no target selected."
}

// `content` is an optional string.
function set(elementId, content) {
    if (elementId in elementIdToDefault) {
        document.getElementById(elementId).innerHTML = (
            content ? content : elementIdToDefault[elementId]
        )
    } else {
        throw new Error()
    }
}

function setElementDefaults() {
    for (let elementId in elementIdToDefault) {
        set(elementId, null)
    }
}

async function loadJson(url) {
    const response = await fetch(url)
    const json = await response.json()
    return json
}

async function main() {
    setElementDefaults()

    set("status", "Loading drugs.")
    const drugIdToProjection = await loadJson(drugProjectionsUrl)
    const drugIds = Object.keys(drugIdToProjection)
    const drugProjections = Object.values(drugIdToProjection)
    set("drugs", drugIds.map(drugId => "<li>".concat(drugId).concat("</li>")).join(""))

    set("status", "Loading targets.")
    const targetIdToProjection = await loadJson(targetProjectionsUrl)
    const targetIds = Object.keys(targetIdToProjection)
    const targetProjections = Object.values(targetIdToProjection)
    set("targets", targetIds.map(targetId => "<li>".concat(targetId).concat("</li>")).join(""))

    set("status", "Loading predictions.")
    const predictions = await loadJson(predictionsUrl)

    set("status", "Plotting drug and target projections.")
    const projections = drugProjections.concat(targetProjections)
    const xBound = Math.max(...projections.map(projection => Math.abs(projection[0]))) * (1 + visualizationPadding)
    const yBound = Math.max(...projections.map(projection => Math.abs(projection[1]))) * (1 + visualizationPadding)
}

main()
