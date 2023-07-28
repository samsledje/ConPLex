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
const visualizationNotSelectedPointOpacity = 0.2

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
            content !== null ? content : elementIdToDefault[elementId]
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

function calculatePositionPercentage(value, bound) {
    return 100 * (value + bound) / (2 * bound)
}

function getDrugPointElementId(drugIndex) {
    return "drug".concat("-").concat(drugIndex.toString())
}

function getTargetPointElementId(targetIndex) {
    return "target".concat("-").concat(targetIndex.toString())
}

function getDrugPointElements(drugIds) {
    return [...drugIds.keys()].map(drugIndex => document.getElementById(getDrugPointElementId(drugIndex)))
}

function getTargetPointElements(targetIds) {
    return [...targetIds.keys()].map(targetIndex => document.getElementById(getTargetPointElementId(targetIndex)))
}

// Optional integers.
var selectedDrugIndex = null;
var selectedTargetIndex = null;

// NOTE: The prediction element should be set after calling these functions.
function toggleSelectedDrug(drugIndex, drugIds, predicitions) {
    drugPointElements = getDrugPointElements(drugIds)
    if (drugIndex === selectedDrugIndex) {
        selectedDrugIndex = null
        set("selected-drug", null)

        for (let otherDrugPointElement of drugPointElements) {
            otherDrugPointElement.style.opacity = null
        }
    } else {
        selectedDrugIndex = drugIndex
        set("selected-drug", drugIds[drugIndex])

        for (let [otherDrugIndex, otherDrugPointElement] of drugPointElements.entries()) {
            if (otherDrugIndex === drugIndex) {
                otherDrugPointElement.style.opacity = null
            } else {
                otherDrugPointElement.style.opacity = visualizationNotSelectedPointOpacity
            }
        }
    }
}

function toggleSelectedTarget(targetIndex, targetIds, predictions) {
    targetPointElements = getTargetPointElements(targetIds)
    if (targetIndex === selectedTargetIndex) {
        selectedTargetIndex = null
        set("selected-target", null)

        for (let otherTargetPointElement of targetPointElements) {
            otherTargetPointElement.style.opacity = null
        }
    } else {
        selectedTargetIndex = targetIndex
        set("selected-target", targetIds[targetIndex])

        for (let [otherTargetIndex, otherTargetPointElement] of targetPointElements.entries()) {
            if (otherTargetIndex === targetIndex) {
                otherTargetPointElement.style.opacity = null
            } else {
                otherTargetPointElement.style.opacity = visualizationNotSelectedPointOpacity
            }
        }
    }
}

function setPrediction(drugIds, targetIds, predictions) {
    if (selectedDrugIndex !== null && selectedTargetIndex !== null) {
        set("prediction", predictions[drugIds[selectedDrugIndex]][targetIds[selectedTargetIndex]].toString())
    } else {
        set("prediction", null)
    }
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

    for (let drugIndex = 0; drugIndex < drugIds.length; drugIndex++) {
        let point = document.createElement("a")
        point.id = getDrugPointElementId(drugIndex)
        point.href = "javascript:;"
        point.onclick = function () {toggleSelectedDrug(drugIndex, drugIds); setPrediction(drugIds, targetIds, predictions)}

        point.classList.add("position-absolute")
        point.classList.add("translate-middle")
        point.classList.add("bg-warning")
        point.classList.add("rounded-circle")

        point.style.left = calculatePositionPercentage(drugProjections[drugIndex][0], xBound).toString().concat("%")
        point.style.top = (100 - calculatePositionPercentage(drugProjections[drugIndex][1], yBound)).toString().concat("%")
        point.style.height = "2%"
        point.style.width = "2%"

        visualizationElement.appendChild(point)
    }

    for (let targetIndex = 0; targetIndex < targetIds.length; targetIndex++) {
        let point = document.createElement("a")
        point.id = getTargetPointElementId(targetIndex)
        point.href = "javascript:;"
        point.onclick = function () {toggleSelectedTarget(targetIndex, targetIds); setPrediction(drugIds, targetIds, predictions)}

        point.classList.add("position-absolute")
        point.classList.add("translate-middle")
        point.classList.add("bg-info")
        point.classList.add("rounded-circle")

        point.style.left = calculatePositionPercentage(targetProjections[targetIndex][0], xBound).toString().concat("%")
        point.style.top = (100 - calculatePositionPercentage(targetProjections[targetIndex][1], yBound)).toString().concat("%")
        point.style.height = "2%"
        point.style.width = "2%"

        visualizationElement.appendChild(point)
    }

    set("status", "Displaying visualization.")
}

main()
