style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Watercolor",
        "prompt": "watercolor painting, {prompt}. vibrant, beautiful, painterly, detailed, textural, artistic",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
    },
    {
        "name": "Film Noir",
        "prompt": "film noir style, ink sketch|vector, {prompt} highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Neon",
        "prompt": "masterpiece painting, buildings in the backdrop, kaleidoscope, lilac orange blue cream fuchsia bright vivid gradient colors, the scene is cinematic, {prompt}, emotional realism, double exposure, watercolor ink pencil, graded wash, color layering, magic realism, figurative painting, intricate motifs, organic tracery, polished",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Jungle",
        "prompt": 'waist-up "{prompt} in a Jungle" by Syd Mead, tangerine cold color palette, muted colors, detailed, 8k,photo r3al,dripping paint,3d toon style,3d style,Movie Still',
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Mars",
        "prompt": "{prompt}, Post-apocalyptic. Mars Colony, Scavengers roam the wastelands searching for valuable resources, rovers, bright morning sunlight shining, (detailed) (intricate) (8k) (HDR) (cinematic lighting) (sharp focus)",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Vibrant Color",
        "prompt": "vibrant colorful, ink sketch|vector|2d colors, at nightfall, sharp focus, {prompt}, highly detailed, sharp focus, the clouds,colorful,ultra sharpness",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Snow",
        "prompt": "cinema 4d render, {prompt}, high contrast, vibrant and saturated, sico style, surrounded by magical glow,floating ice shards, snow crystals, cold, windy background, frozen natural landscape in background  cinematic atmosphere,highly detailed, sharp focus, intricate design, 3d, unreal engine, octane render, CG best quality, highres, photorealistic, dramatic lighting, artstation, concept art, cinematic, epic Steven Spielberg movie still, sharp focus, smoke, sparks, art by pascal blanche and greg rutkowski and repin, trending on artstation, hyperrealism painting, matte painting, 4k resolution",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
    },
    {
        "name": "Line art",
        "prompt": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
    },
        {
        "name": "Art Nouveau",
        "prompt": "Art Nouveau style, {prompt}, organic forms, curvilinear lines, elegant, nature-inspired, intricate patterns, flowing designs, soft colors",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, brutalism, geometric, harsh lines, noisy",
    },
    {
        "name": "Cubism",
        "prompt": "cubist painting, {prompt}, abstract, geometric shapes, fragmented objects, multiple viewpoints, bold lines, reduced colors, Pablo Picasso inspired",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealistic, smooth, flowing, noisy",
    },
    {
        "name": "Minimalism",
        "prompt": "minimalist art, {prompt}, simple, clean lines, monochrome, negative space, minimal detail, geometric shapes, modern, sophisticated",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, baroque, cluttered, over-detailed, noisy",
    },
    {
        "name": "Baroque",
        "prompt": "baroque style, {prompt}, grandeur, drama, contrast, rich colors, intense lighting, ornate, Caravaggio inspired, emotional, detailed",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, minimalist, flat, dull, noisy",
    },
    {
        "name": "Expressionism",
        "prompt": "expressionist art, {prompt}, emotional, distorted, exaggerated, vivid colors, dynamic brushstrokes, subjective perspective, impactful",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealism, bland, underwhelming, noisy",
    },
    {
        "name": "Digital Glitch",
        "prompt": "digital glitch art, {prompt}, corrupted image, tech-inspired, vibrant, surreal, abstract, pixelated, data moshing, futuristic",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, classical, realistic, smooth, noiseless",
    },
    {
        "name": "Psychedelic",
        "prompt": "psychedelic art, {prompt}, vivid colors, hallucinatory patterns, surreal, fluid shapes, trippy, 1960s style, kaleidoscopic, vibrant",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, monochrome, simplistic, noiseless",
    },
    {
        "name": "Victorian",
        "prompt": "Victorian style, {prompt}, elegant, ornate, romantic, historical, detailed patterns, rich textures, sophisticated, vintage",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, modern, minimalist, plain, noisy",
    },
    {
        "name": "Graffiti",
        "prompt": "graffiti art, {prompt}, urban, street style, bold colors, spray paint, tagging, hip-hop culture, dynamic, expressive, contemporary",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, classical, fine art, noiseless, smooth",
    },
    {
        "name": "Ukiyo-e",
        "prompt": "Ukiyo-e style, {prompt}, Japanese woodblock print, flat color areas, bold outlines, historical scenes, nature, Edo period, traditional",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, 3D, photorealistic, modern, noisy",
    },
        {
        "name": "Retro Futurism",
        "prompt": "retro futurism, {prompt}, vibrant colors, geometric shapes, streamline design, 1950s and 1960s style, optimistic, space-age, visionary, neon lighting, bold typography",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, medieval, deformed, glitch, blurry, noisy",
    },
    {
        "name": "Steampunk",
        "prompt": "steampunk style, {prompt}, Victorian era, industrial, mechanical gears, brass and copper, steam engines, intricate details, fantastical machines, sepia tones",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, neon, futuristic, blurry, noisy",
    },
    {
        "name": "Cyberpunk",
        "prompt": "cyberpunk aesthetic, {prompt}, neon lights, urban dystopia, futuristic, high-tech, low-life, cybernetics, dark and gritty, vivid colors, digital world",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, medieval, steampunk, blurry, noisy",
    },
    {
        "name": "Impressionist",
        "prompt": "impressionist style painting, {prompt}, vibrant, light brushstrokes, open composition, emphasis on light in its changing qualities, movement, ordinary subject matter",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
    },
    {
        "name": "Art Deco",
        "prompt": "art deco style, {prompt}, glamorous, elegant, functional, geometric patterns, bold colors, sleek lines, chrome, glass, shiny fabrics",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, rustic, medieval, deformed, glitch, noisy",
    },
    {
        "name": "Fantasy",
        "prompt": "fantasy style, {prompt}, mythical creatures, enchanted forests, magic elements, dreamlike landscapes, vibrant colors, detailed, imaginative",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
    },
    {
        "name": "Gothic",
        "prompt": "gothic style, {prompt}, dark and moody, gothic architecture, medieval elements, dramatic lighting, somber tones, intricate details, mysterious atmosphere",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, neon, futuristic, blurry, noisy",
    },
    {
        "name": "Pop Art",
        "prompt": "pop art style, {prompt}, bold colors, mass culture, comic style, ironic, whimsical, repetition of images, bright, high contrast",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, medieval, steampunk, blurry, noisy",
    },
    {
        "name": "Surrealism",
        "prompt": "surrealist style, {prompt}, dreamlike, bizarre, irrational, abstract, imaginative, distorted reality, vivid, unexpected juxtapositions",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
    },
    {
        "name": "Abstract",
        "prompt": "abstract style, {prompt}, non-representational, shapes, forms, colors, lines, dynamic, modern, expressive, non-figurative, bold",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}