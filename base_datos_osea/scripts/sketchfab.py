"""Cataloga y descarga mallas 3D de Sketchfab, curadas a mano.

A diferencia de MorphoSource/Smithsonian, aquí NO hay búsqueda+selección
automática: el fémur `000031310` de MorphoSource resultó ser un fragmento
(le faltaban los cóndilos distales) etiquetado como "femur" sin ninguna
palabra como "proximal"/"fragmento" que un clasificador de texto pudiera
pillar. Sketchfab permite ver una miniatura de cada modelo ANTES de
descargarlo — así que cada candidato de este fichero fue revisado
visualmente (por Claude, mirando la miniatura) antes de añadirlo aquí.
Añadir un candidato nuevo: buscar en https://sketchfab.com/developers/data-api/v3
(`/v3/search?type=models&q=...&downloadable=true`), mirar su miniatura, y
solo si se ve la pieza completa añadirlo a CURATED_CANDIDATES.

La búsqueda y las miniaturas son públicas (sin token). La descarga requiere
un token de cuenta gratuita de Sketchfab (Settings > Password & API).

Uso:
    python scripts/sketchfab.py catalog
    python scripts/sketchfab.py download
"""

import argparse
import csv
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "catalog" / "sketchfab_media.csv"
MESHES_DIR = REPO_ROOT / "data" / "meshes"
API_URL = "https://api.sketchfab.com/v3"

# Cada entrada fue revisada visualmente (miniatura) antes de añadirse — ver
# docstring. `species` por defecto Homo sapiens salvo que se indique otra.
CURATED_CANDIDATES = [
    {
        "bone_canonical": "femur",
        "uid": "a9c1f1a88b104c3fbfe975fa10b31b31",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cabeza + diáfisis + cóndilos distales). Elon University anatomy lab.",
    },
    {
        "bone_canonical": "humerus",
        "uid": "12744d7c36be47d8ad94f620a22d649c",
        "species": "Homo sapiens",
        "notes": "Verificado completo (ambas epífisis visibles). CC0.",
    },
    {
        "bone_canonical": "cranium",
        "uid": "c39855833ff74943a2847972c9ff8700",
        "species": "Homo sapiens",
        "notes": "Verificado completo con mandíbula visible. Fotogrametría documentada, UNC Greensboro, permiso citado.",
    },
    {
        "bone_canonical": "mandible",
        "uid": "7a3c3b99916e481197e6f18fda107c15",
        "species": "Homo sapiens",
        "notes": "Verificado completo (ambas ramas con cóndilos, dientes y textura). University of Dundee, creadores documentados.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "24d2b770d5e049e08b4239dac463de39",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cuerpo, glenoides, acromion). UNCG Imaging Lab, escaneo NextEngine de espécimen de colección con permiso citado.",
    },
    {
        "bone_canonical": "clavicle",
        "uid": "b58db46ed8564fedade966310a3e95ae",
        "species": "Homo sapiens",
        "notes": "Verificado completo (ambos extremos, curva en S). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "radius",
        "uid": "68c3f1c56b1b400e905072293e0a878a",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cabeza radial proximal + apófisis estiloides distal). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "ulna",
        "uid": "4b849ef3f5c142099b0a244b00011e47",
        "species": "Homo sapiens",
        "notes": "Verificado completo (olécranon proximal + apófisis estiloides distal). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "tibia",
        "uid": "7c1979d6127749bc80a9d9276d24edcd",
        "species": "Homo sapiens",
        "notes": "Verificado completo (mesetas tibiales proximales + maléolo medial distal). Eric Bauer, alta resolución.",
    },
    {
        "bone_canonical": "fibula",
        "uid": "5f239617db8d4d2eaaf13b8616acf70e",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cabeza proximal + maléolo lateral distal). Eric Bauer.",
    },
    {
        "bone_canonical": "patella",
        "uid": "8459a5c2baee48c8905631322cdf8edf",
        "species": "Homo sapiens",
        "notes": "Verificado completo (forma triangular característica). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "3f7cfb377bb74b49aaf3a98e8933e94b",
        "species": "Homo sapiens",
        "notes": "Verificado completo (forma triangular, forámenes sacros visibles). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "da43de500e614feea3458da2e6bc8e5f",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cuerpo, apófisis espinosa y transversas, pedículos). UNCG Imaging Lab, lumbar.",
    },
    {
        "bone_canonical": "rib",
        "uid": "6ca3afdf9a084b0db8b6141f50a94f49",
        "species": "Homo sapiens",
        "notes": "Verificado completo (costilla central típica, cabeza a un extremo). UNCG Imaging Lab.",
    },
    # --- 2º espécimen para los 2 grupos morfológicos que solo tenían 1 (ver
    # BONE_GROUPS en bones.py): sin esto, el clasificador de grupo no puede
    # distinguir "forma del hueso" de "forma de ESTE hueso concreto" para
    # hueso_plano/pelvis.
    {
        "bone_canonical": "scapula",
        "uid": "0745bbb368b4401db89e73babe440ee8",
        "species": "Homo sapiens",
        "notes": "2º espécimen (grupo hueso_plano tenía solo 1). Verificado completo (glenoides, acromion, coracoides). Eric Bauer, alta resolución (9.8M caras).",
    },
    {
        "bone_canonical": "pelvis",
        "uid": "35586f343d9c4c6eb813f9006f036595",
        "species": "Homo sapiens",
        "notes": "2º espécimen (grupo pelvis tenía solo 1). Verificado completo (ambos ilion + sacro, forámenes visibles). Oregon State University.",
    },
    # --- 3er espécimen para costilla/sacro/vértebra: con GroupKFold (espécimen
    # completo fuera) estos 3 grupos caían a 0% de recall en algunos folds al
    # tener solo 2 especímenes — con 2 no hay margen para que el modelo vea la
    # clase en train Y en test a la vez en una partición de 5 folds.
    {
        "bone_canonical": "rib",
        "uid": "4d0e5894afdd4a0b8a327c45e905859f",
        "species": "Homo sapiens",
        "notes": "3er espécimen (grupo costilla tenía solo 2). Duodécima costilla — forma muy distinta (corta, poco curvada) a la costilla central ya usada, buena diversidad de forma dentro de la clase. UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "c606367c21f64fe6b6d6f8f2b3ec4da4",
        "species": "Homo sapiens",
        "notes": "3er espécimen (grupo sacro tenía solo 2). Sacro con cóccix articulado, forámenes sacros visibles. Autor independiente (Ualde), no UNCG/Eric Bauer.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "abd1b3a2116a449f94bccb86d1e17f54",
        "species": "Homo sapiens",
        "notes": "3er espécimen (grupo vertebra tenía solo 2). Vértebra torácica (T5/T6) completa, cuerpo+apófisis espinosa+transversas+carillas. Eric Bauer, alta resolución.",
    },
    # --- Ronda 2: empujar costilla/sacro/vertebra/hueso_plano hacia 6+
    # especímenes (con 3 seguían sin generalizar de forma estable en
    # GroupKFold repetido — ver HISTORIAL.md). Pelvis se queda en 2: no se
    # encontró más modelos de pelvis bilateral completa que no llevaran
    # fémures/vértebras pegados (rechazados por mezclar huesos en una malla).
    {
        "bone_canonical": "rib",
        "uid": "6d4c1e0e0bc042ec8c36d3ec16ccaf62",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Segunda costilla izquierda, completa. UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "rib",
        "uid": "6e9f7dc6d7cb4b9ab343ca95ea3459d2",
        "species": "Homo sapiens",
        "notes": "5º espécimen. Primera costilla izquierda (forma ancha/aplanada característica), completa. UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "c7628ff7a69643af9dc7493f799a68e2",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Completo, forámenes visibles. Andreas Kontny, autor independiente.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "654aceb8ffd44e0896760d4ed8465dd8",
        "species": "Homo sapiens",
        "notes": "5º espécimen. Completo (algo erosionado, espécimen real), forámenes visibles.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "5c725158053a4e26adc2d283175cdf38",
        "species": "Homo sapiens",
        "notes": "6º espécimen. Espécimen arqueológico documentado (St. Nicolas Kirk SK697), completo con forámenes. CC-BY-NoDerivs.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "29b6bb536df84df08fa0ad444c3b01d1",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Vértebra cervical (C3/C4), completa. Eric Bauer.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "03657338956d41158be94cf19b150402",
        "species": "Homo sapiens",
        "notes": "5º espécimen. Atlas (C1) — forma de anillo muy distinta al resto (sin cuerpo vertebral), buena diversidad de forma dentro de la clase. Eric Bauer.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "7c5e902085294a6fa9fda5cc775e0e17",
        "species": "Homo sapiens",
        "notes": "6º espécimen. Cuarta vértebra cervical, completa. UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "da38f448a06749daa09859d767971088",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Completo (glenoides, acromion, coracoides), baja resolución pero forma correcta.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "3a749164f2ef479093cc5e92abefec0d",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Completo, alta resolución. Eric Bauer (espécimen distinto al ya usado del mismo autor). CC-BY-NC-ND.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "33b0fbce3b634fa7bba801b142b9aeb6",
        "species": "Homo sapiens",
        "notes": "5º espécimen. Completo. Ualde, autor independiente.",
    },

    # --- Ronda 3 (2026-07-24): ampliación grande tras el banco de evaluación fijo revelar
    # que la mayoría de huesos tenían 1 solo espécimen físico por especie (o por hueso, en
    # hueso_largo/hueso_pequeno) — GroupKFold con "especimen fuera" es en la práctica
    # "especie fuera" con n=1. Buscado por 4 agentes en paralelo, cada candidato verificado
    # visualmente por miniatura antes de añadirse aquí, igual que las rondas anteriores.
    # --- hueso_largo: profundizar femur/humerus/radius/ulna/tibia/fibula/clavicle (todos
    # tenían solo 1 espécimen sapiens antes de esta ronda) ---
    {
        "bone_canonical": "femur",
        "uid": "ebeaa853101a46529556ac7782d23296",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (cabeza/cuello proximal + cóndilos distales). Ualde, autor independiente.",
    },
    {
        "bone_canonical": "femur",
        "uid": "0647ff9a00924f40a06e08361cea176c",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (cabeza/trocánter proximal + cóndilos distales). Tom O'Graphyx.",
    },
    {
        "bone_canonical": "femur",
        "uid": "3fbe6a09153b4a0cb38d15005726d8ff",
        "species": "Gorilla gorilla",
        "notes": "Diversidad de especie. Verificado completo (cabeza + cóndilos). Espécimen 'Samson', gorila macho documentado (New York State Museum, ZM 3922). Virtual Curation Lab. CC-BY-NC-ND.",
    },
    {
        "bone_canonical": "humerus",
        "uid": "548132dc5a0a423581e2cba2014ea521",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (cabeza + tróclea/cóndilo distal). Eric Bauer, Elon University. CC-BY-NC-ND.",
    },
    {
        "bone_canonical": "humerus",
        "uid": "ba96c6f7caba4a9b9958210ae5e5f78c",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (cabeza + epicóndilos distales). Andreas Kontny / Medical University Varna, Bulgaria.",
    },
    {
        "bone_canonical": "humerus",
        "uid": "72442a01b58c49c48ca6d4854f5e8bde",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Verificado completo (cabeza + tróclea distal). UNCG Imaging Lab, NextEngine scanner, permiso Dr. Robert Anemone.",
    },
    {
        "bone_canonical": "humerus",
        "uid": "ec66ba21bfdc4a4191dd4d73239c910a",
        "species": "Homo sapiens",
        "notes": "5º espécimen. Verificado completo (cabeza + tróclea distal). 'Wisad Humerus 1', ECU Bioarchaeology Laboratory, tumba jordana s. IV-VI, préstamo documentado.",
    },
    {
        "bone_canonical": "radius",
        "uid": "54bd4f2bbf40423693d954d4461e2dc2",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (cabeza radial + apófisis estiloides). Dr. Alex Cheroske, alta resolución (2.4M caras).",
    },
    {
        "bone_canonical": "radius",
        "uid": "0ffd5799d12f47a08b858687e405f505",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (cabeza + estiloides distal). Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "radius",
        "uid": "e6c60b21485d44358114fa8210a7ba8c",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Verificado completo (miniatura algo desenfocada pero ambas epífisis reconocibles). Virtual Curation Lab, espécimen docente VCU. CC-BY-NC-ND.",
    },
    {
        "bone_canonical": "radius",
        "uid": "5859fa73306d45fe96049d60468ec996",
        "species": "Homo sapiens",
        "notes": "5º espécimen. Verificado completo (cabeza + estiloides). Eric Bauer, Elon University.",
    },
    {
        "bone_canonical": "ulna",
        "uid": "cd6a2f43969f4153ad395442daa86503",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (olécranon + apófisis estiloides). Eric Bauer, Elon University.",
    },
    {
        "bone_canonical": "ulna",
        "uid": "90376c72e00942b38c7fc87364d20e3c",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (olécranon + estiloides). Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "tibia",
        "uid": "40d17b94e8fe491e8cf0df6a3bfa420f",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (meseta tibial + maléolo medial). Autor 'fedorageevsigmachel', alta resolución (2.9M caras).",
    },
    {
        "bone_canonical": "tibia",
        "uid": "e00a9d46dda7432787d14a794b23a0c3",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (ambos extremos presentes), con lesión ósea patológica real visible que no afecta la forma general. Swedish History Museum (Historiska), espécimen arqueológico documentado.",
    },
    {
        "bone_canonical": "tibia",
        "uid": "f8c5e275345c47a3b0989fa27885812e",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Verificado completo (meseta + maléolo medial). Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "fibula",
        "uid": "c1323d5bf57d48a9929304635663a41a",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (cabeza proximal + maléolo lateral). Autor 'fedorageevsigmachel'.",
    },
    {
        "bone_canonical": "fibula",
        "uid": "953e1ef6a7cd49468550032cdc6984d5",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (cabeza + maléolo lateral). Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "clavicle",
        "uid": "fd3ae947be90403d9a79d8e9669af236",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (curva en S, ambos extremos). Eric Bauer, Elon University, alta resolución (2.4M caras).",
    },
    {
        "bone_canonical": "clavicle",
        "uid": "6e0e3695676e442c90efcdf71e491682",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (curva en S, ambos extremos). Andreas Kontny / Medical University Varna.",
    },

    # --- cranio/mandibula_maxilar: profundizar sapiens + 2 especies de primate nuevas
    # (maxilar sigue sin reemplazo fiable tras una 2ª búsqueda, ver HISTORIAL.md) ---
    {
        "bone_canonical": "cranium",
        "uid": "78ee6710760241d19920a3a36b97ff31",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (bóveda + órbitas + maxilar con dientes visibles). San Antonio College, colección de antropología, escaneo Artec Eva.",
    },
    {
        "bone_canonical": "cranium",
        "uid": "32a69b1fc0ba45a2ac2d91889657407f",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo en vista frontal (bóveda + órbitas + nasal + maxilar con dientes); base craneal no confirmada visualmente, solo se roto miniatura frontal. Segmentación de CT-scan de persona viva (jagake).",
    },
    {
        "bone_canonical": "cranium",
        "uid": "c33d30ae8272473ca8f648cf2c31956b",
        "species": "Pongo sp.",
        "notes": "Especie nueva (orangután, género no cubierto). Verificado completo (bóveda + cresta sagital + mandíbula articulada + dentición completa). VISIBLE APE PROJECT.",
    },
    {
        "bone_canonical": "cranium",
        "uid": "f90d811c1fd047e486123d4e0437aed5",
        "species": "Macaca sylvanus",
        "notes": "Especie nueva (macaco de Berbería, distinto de macaca_radiata ya cubierto). Verificado completo (cráneo + mandíbula articulada con boca abierta, dentición completa). Yale Peabody Museum, YPM MAM.006878.",
    },
    {
        "bone_canonical": "mandible",
        "uid": "cb5f3a27ba924a26b86a073178179fa8",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (ambas ramas con cóndilos, dentición mixta). Melbourne Dental School, escaneo David SLS-4.",
    },
    {
        "bone_canonical": "mandible",
        "uid": "acbe657d10164192b85a5737c41064d4",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (ambas ramas con cóndilos, dientes, alta definición). gwuanthro (George Washington University Anthropology).",
    },
    {
        "bone_canonical": "mandible",
        "uid": "2667ab0d436840f58a82da7cefb8b5be",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Verificado completo (ambas ramas con cóndilos y dientes). Segmentación de CT-scan de persona viva (jagake), pareja del cranium jagake de arriba.",
    },

    # --- hueso_pequeno/hueso_plano/pelvis: profundizar los que solo tenían 1 ejemplar ---
    {
        "bone_canonical": "patella",
        "uid": "c24c83826aa945958736a0ed73787967",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (superficie articular posterior visible). Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "metacarpal_1",
        "uid": "153fe961b2cc40dd85357f33ca97d7c6",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo (base + cabeza). Cementerio medieval de Valeni (Patakfalva), Rumanía, permiso Haáz Rezső Múzeum. USF IDEx.",
    },
    {
        "bone_canonical": "metacarpal_1",
        "uid": "8936457b2a94406c9d5050c6421a73bb",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (base + cabeza). Oregon State University (osuecampus).",
    },
    {
        "bone_canonical": "metatarsal_1",
        "uid": "595c2cac5c0f49a68f2dc358d8891026",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo. Valeni/Patakfalva, Rumanía. USF IDEx.",
    },
    {
        "bone_canonical": "metatarsal_1",
        "uid": "67c008d209b34902bd5eec768e680b74",
        "species": "Homo sapiens",
        "notes": "3er espécimen, individuo distinto. Verificado completo. Mismo yacimiento. USF IDEx.",
    },
    {
        "bone_canonical": "metatarsal_1",
        "uid": "435864b65aa24be78f9294300881d218",
        "species": "Homo sapiens",
        "notes": "4º espécimen, individuo distinto. Verificado completo. Mismo yacimiento. USF IDEx.",
    },
    {
        "bone_canonical": "metatarsal_1",
        "uid": "9463fbeb93584d6cae99f468093c0cbd",
        "species": "Homo sapiens",
        "notes": "5º espécimen, individuo distinto, baja resolución pero forma correcta. Mismo yacimiento. USF IDEx.",
    },
    {
        "bone_canonical": "metatarsal_1",
        "uid": "ea217e8618ff4ee89b6a684ead3cec1d",
        "species": "Homo sapiens",
        "notes": "6º espécimen. Verificado completo. Oregon State University (osuecampus), fuente distinta.",
    },
    {
        "bone_canonical": "trapezium",
        "uid": "8943660fd24e4e709ee4d86926baedca",
        "species": "Homo sapiens",
        "notes": "2º espécimen. Verificado completo. Eric Bauer, Elon University.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "c14501f6adf345e8a7e3ba22d485d4f6",
        "species": "Homo sapiens",
        "notes": "6º espécimen. Escápula izquierda completa. Cisterna Qasr Hallabat, Jordania (s. VIII-X d.C.), préstamo documentado. ECU Bioarchaeology Laboratory.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "22e60495b7984dfa88d7198267512cb9",
        "species": "Homo sapiens",
        "notes": "7º espécimen, individuo distinto (misma cisterna). ECU Bioarchaeology Laboratory.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "8e26c84bfd544618b48c0306e8780d14",
        "species": "Homo sapiens",
        "notes": "8º espécimen. Escápula derecha completa, individuo distinto (misma cisterna). ECU Bioarchaeology Laboratory.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "3b89ace6671144a5ac05a84cc38843da",
        "species": "Homo sapiens",
        "notes": "9º espécimen. Escápula derecha completa, procedencia distinta (tumba s. IV-VI, Wisad Pools, Jordania). ECU Bioarchaeology Laboratory.",
    },
    {
        "bone_canonical": "pelvis",
        "uid": "55e3c54dc2e04d23b0cf46da58c71994",
        "species": "Homo sapiens",
        "notes": "3er espécimen. Verificado completo (ambos ilion + isquion + pubis + sacro con forámenes, sin huesos extra). Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "pelvis",
        "uid": "10f950afa4284019a1d744ac14b6cf90",
        "species": "Homo sapiens",
        "notes": "4º espécimen. Verificado completo (anillo pélvico bilateral íntegro con sacro, sínfisis púbica visible). Ualde, autor independiente.",
    },

    # --- sacro/vertebra/costilla: los 3 grupos peor evaluados hoy en el banco fijo
    # (0-25% recall) por depender del único espécimen naledi, reservado para el banco,
    # como su único ejemplo "distinto" — con esto dejan de depender de él ---
    {
        "bone_canonical": "sacrum",
        "uid": "49aafdac520046ab97afc33a26d47d0c",
        "species": "Homo sapiens",
        "notes": "7º espécimen. Verificado completo (forámenes + cóccix articulado). Eric Bauer.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "eecdb1ffbeca440b87cb3ca9760e7dc8",
        "species": "Homo sapiens",
        "notes": "8º espécimen. Verificado completo (8 forámenes simétricos, cóccix visible). Team Skeleton Wars.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "5311d770cde4466082838785d1120e8a",
        "species": "Homo sapiens",
        "notes": "9º espécimen. Sacro arqueológico (Elbolton Cave, North Yorkshire, Craven Museum & Gallery), completo, forámenes visibles.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "20b7c2d45d7f4e29a071f616906fe895",
        "species": "Homo sapiens",
        "notes": "10º espécimen. Verificado completo (alas, forámenes, cóccix). Oregon State University eCampus.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "2eeac947e7a04360b5f044efe54d19ee",
        "species": "Homo sapiens",
        "notes": "11º espécimen. Verificado completo (forámenes claros, alas anchas). Oregon State University eCampus, escaneo distinto al anterior.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "34eda94dbc264baab7757eb34e1f9b04",
        "species": "Homo sapiens",
        "notes": "7º espécimen. Axis (C2) completo, odontoides + forámenes transversos + apófisis espinosa. Eric Bauer.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "3d126cced9fd4f8fb327865a0a79dace",
        "species": "Homo sapiens",
        "notes": "8º espécimen. Atlas (C1) completo, 2º atlas de especimen distinto al ya usado. UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "f70c894e0d9440b9a77861d278e1c409",
        "species": "Homo sapiens",
        "notes": "9º espécimen. Axis (C2) completo, cuerpo+dens+carillas. UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "26187e5e5a05475691e8ebf73e880546",
        "species": "Homo sapiens",
        "notes": "10º espécimen. L5 completa. UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "459be5e9970746a89ec4482daca1f55c",
        "species": "Homo sapiens",
        "notes": "11º espécimen. T1 completa. Oregon State University eCampus.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "a1ae4d49c9f54e7395f803c6c013fd17",
        "species": "Homo sapiens",
        "notes": "12º espécimen. T12 (forma de transición hacia lumbar). Oregon State University eCampus.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "37fa466e6a7f4d2d9f128541a3a16e69",
        "species": "Homo sapiens",
        "notes": "13º espécimen. C7 ('vertebra prominens'), apófisis espinosa larga característica. Oregon State University eCampus.",
    },
    {
        "bone_canonical": "rib",
        "uid": "8798b268c39748f29ef7b4ca150da091",
        "species": "Homo sapiens",
        "notes": "6º espécimen. Primera costilla, completa. Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "rib",
        "uid": "cb2f2ea263024913afc76e418c766ea9",
        "species": "Homo sapiens",
        "notes": "7º espécimen. Última costilla (12ª, flotante), completa. Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "rib",
        "uid": "40f79aee5a9f4099b8c81a3266976893",
        "species": "Homo sapiens",
        "notes": "8º espécimen. Costa VIII (forma típica central), completa. Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "rib",
        "uid": "b2a9d0876d2b410c80ac334bf9426e33",
        "species": "Homo sapiens",
        "notes": "9º espécimen. Costa I completa. Andreas Kontny / Medical University Varna.",
    },
    {
        "bone_canonical": "rib",
        "uid": "cf67ae5ec5df432883667c34f29f7d6e",
        "species": "Homo sapiens",
        "notes": "10º espécimen. UWF5 First Rib, completa. Mariana Z, University of West Florida (individuo forense distinto a Varna). CC-BY-NC.",
    },
    {
        "bone_canonical": "rib",
        "uid": "5778cda329b54eb5865270739b4463bb",
        "species": "Homo sapiens",
        "notes": "11º espécimen. UWF8 Rib 4, completa. Mariana Z, University of West Florida. CC-BY-NC.",
    },
]

# Maxilar: 2 candidatos revisados (uid 47125f083e464d87b23ec748b2679983 y
# e59a4be88e8545268dda24c72eed2a0c), ambos fragmentos unilaterales sin ambos
# lados de la arcada — rechazados. Sigue cubierto por H. naledi.

FIELDS = [
    "bone_canonical", "species", "uid", "name", "license", "verified_complete", "verification_notes",
    "face_count", "creator", "description", "thumbnail_url", "viewer_url",
    "download_status", "local_path", "downloaded_at",
]


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_existing(path):
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing[row["uid"]] = row
    return existing


def build_catalog():
    existing = load_existing(CATALOG_PATH)
    rows = []
    for candidate in CURATED_CANDIDATES:
        resp = requests.get(f"{API_URL}/models/{candidate['uid']}")
        resp.raise_for_status()
        data = resp.json()
        prev = existing.get(candidate["uid"], {})
        thumb = max(data["thumbnails"]["images"], key=lambda i: i["width"])
        rows.append({
            "bone_canonical": candidate["bone_canonical"],
            "species": candidate["species"],
            "uid": candidate["uid"],
            "name": data["name"],
            "license": data["license"]["label"],
            "verified_complete": True,
            "verification_notes": candidate["notes"],
            "face_count": data.get("faceCount", ""),
            "creator": data["user"]["displayName"],
            "description": (data.get("description") or "").replace("\n", " ").strip(),
            "thumbnail_url": thumb["url"],
            "viewer_url": data["viewerUrl"],
            "download_status": prev.get("download_status", ""),
            "local_path": prev.get("local_path", ""),
            "downloaded_at": prev.get("downloaded_at", ""),
        })

    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CATALOG_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"=== Catálogo Sketchfab ({len(rows)} candidatos curados a mano) ===")
    for row in rows:
        print(f"  {row['bone_canonical']:<10} {row['name']:<30} {row['license']:<12} {row['uid']}")
    print(f"\nCSV escrito en: {CATALOG_PATH}")


def download_catalog():
    if not CATALOG_PATH.exists():
        sys.exit(f"No existe {CATALOG_PATH}. Ejecuta antes: python scripts/sketchfab.py catalog")

    token = os.environ.get("SKETCHFAB_API_TOKEN")
    if not token:
        sys.exit("Falta SKETCHFAB_API_TOKEN en el entorno o en .env.")
    headers = {"Authorization": f"Token {token}"}

    with CATALOG_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())

    pending = [r for r in rows if r["download_status"] != "done"]
    print(f"{len(pending)} elemento(s) por descargar")
    for row in pending:
        print(f"  -> {row['bone_canonical']} ({row['uid']}) ... ", end="", flush=True)
        try:
            resp = requests.get(f"{API_URL}/models/{row['uid']}/download", headers=headers)
            resp.raise_for_status()
            archives = resp.json()
            archive = archives.get("glb") or archives.get("gltf") or archives.get("source")
            if not archive:
                raise RuntimeError(f"sin archivo descargable en {list(archives.keys())}")

            species_slug = row["species"].lower().replace(" ", "_")
            bone_dir = MESHES_DIR / species_slug / row["bone_canonical"] / f"sketchfab_{row['uid']}"
            bone_dir.mkdir(parents=True, exist_ok=True)
            filename = archive["url"].split("?")[0].rsplit("/", 1)[-1]
            dest = bone_dir / filename

            with requests.get(archive["url"], stream=True) as file_resp:
                file_resp.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in file_resp.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)

            if dest.suffix == ".zip":
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(bone_dir)

            row["download_status"] = "done"
            row["local_path"] = str(bone_dir.relative_to(REPO_ROOT))
            row["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            print("OK")
        except Exception as exc:
            row["download_status"] = f"error: {exc}"
            print(f"ERROR: {exc}")

        with CATALOG_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nCatálogo actualizado: {CATALOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog", "download"])
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if args.command == "catalog":
        build_catalog()
    else:
        download_catalog()


if __name__ == "__main__":
    main()
