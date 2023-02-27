const container = document.getElementById("model-card-container");
const imgPath = "./imgs/"

function modelCard(name) {
    // card div
    let card = document.createElement("div")
    card.className = "model-card"

    // card image div
    let modelImg = document.createElement("div")
    modelImg.className = "model-card-img"

    let Img = document.createElement("img")
    Img.className = "model-card-img-pic"
    Img.src = imgPath + name + (".png")

    modelImg.appendChild(Img)

    // card description div
    let modelDescription = document.createElement("div")
    modelDescription.className = "model-card-description"

    let modelInfo = document.createElement("div")
    modelInfo.className = "model-card-description-info"

    let modelInfoText = document.createElement("p")
    modelInfoText.className = "model-card-description-info-text"
    modelInfoText.innerText = name

    modelInfo.appendChild(modelInfoText)
    modelDescription.appendChild(modelInfo)

    // card overlay
    let modelOverlay = document.createElement("div")
    modelOverlay.className = "model-card-overlay"

    // construct
    card.appendChild(modelImg)
    card.appendChild(modelOverlay)
    card.appendChild(modelDescription)

    // add to container
    container.appendChild(card)


    // card.innerText = "Hello"
    // card.style.cssText = "border:1px; border-style:solid; border-color:white"
    // container.appendChild(card)
    
}

modelCard("Michelle")
modelCard("Vanguard")
modelCard("Josh")
modelCard("Pete")
modelCard("Erika")

modelCard("Jolleen")
modelCard("Mousey")
modelCard("Aj")
// modelCard("Marker")
modelCard("Bryce")
modelCard("Sophie")