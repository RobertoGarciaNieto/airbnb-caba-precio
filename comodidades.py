"""Comodidades de "Probá el modelo".

MODELO: las 6 columnas has_* que usa el modelo, cada una con su casilla.
EXTRAS: las comodidades más frecuentes del snapshot (agrupando variantes como "Wifi – 16 Mbps" en "Wifi") que NO
        activan ninguna columna has_* según las reglas del pipeline (pipeline/dags/airbnb_ingest_V3.py,
        refine_silver). Solo suman a amenities_count.
amenities_count = casillas marcadas (MODELO + EXTRAS), igual que el pipeline: el largo de la lista de amenities.
Entre las dos hay 123 casillas, más que el máximo de cualquier publicación (112).
"""
import re

# Reglas del pipeline (copiadas de refine_silver) para las columnas has_* que usa el modelo.
REGLAS_MODELO = {
    "has_ac": (r"\bair conditioning\b|\bac\b|aire acondicionado", None, "Aire acondicionado"),
    "has_pool": (r"\bpool\b|pileta", r"pool table", "Pileta"),
    "has_elevator": (r"\belevator\b|ascensor", None, "Ascensor"),
    "has_balcony_patio": (r"balcony|patio|terrace|terraza|outdoor", None, "Balcón o patio"),
    "has_parking": (r"\bparking\b|garage|cochera", None, "Cochera"),
    "has_workspace": (r"dedicated workspace|escritorio", None, "Espacio de trabajo"),
}

# (columna, nombre, marcada por defecto, ayuda). Por defecto, las que tiene al menos el 45 % de las publicaciones.
MODELO = [
    ("has_ac", "Aire acondicionado", True, "Aire acondicionado en la unidad (split, central o de ventana)."),
    ("has_pool", "Pileta", False, "Pileta propia o del edificio. Por la hipótesis H4, su efecto depende del barrio."),
    ("has_elevator", "Ascensor", True, "El edificio tiene ascensor."),
    ("has_balcony_patio", "Balcón o patio", False, "Balcón, terraza, patio o espacio al aire libre del huésped."),
    ("has_parking", "Cochera", False, "Estacionamiento: cochera en el edificio o lugar cerca (gratis o pago)."),
    ("has_workspace", "Espacio de trabajo", True, "Escritorio o lugar preparado para trabajar con computadora."),
]
MODELO = [(c, n, d, a + " Es una columna del modelo y además suma 1 a la cantidad de comodidades.")
          for c, n, d, a in MODELO]

# (nombre en Inside Airbnb, nombre en español, marcada por defecto)
# Por defecto: las que tiene al menos el 45 % de las publicaciones (31 casillas; la mediana real es 34).
CATEGORIAS = {
    "Cocina": [
        ("Kitchen", "Cocina", True), ("Dishes and silverware", "Vajilla y cubiertos", True),
        ("Cooking basics", "Elementos básicos de cocina", True), ("Refrigerator", "Heladera", True),
        ("Microwave", "Microondas", True), ("Coffee maker", "Cafetera", True),
        ("Hot water kettle", "Pava eléctrica", True), ("Toaster", "Tostadora", True),
        ("Dining table", "Mesa de comedor", True), ("Wine glasses", "Copas de vino", True),
        ("Freezer", "Freezer", True), ("Oven", "Horno", False), ("Baking sheet", "Placa para horno", False),
        ("Coffee", "Café", False), ("Stove", "Cocina u anafe", False), ("Blender", "Licuadora", False),
        ("Mini fridge", "Heladera chica", False), ("Dishwasher", "Lavavajillas", False),
        ("Trash compactor", "Compactador de basura", False), ("Stainless steel oven", "Horno de acero inoxidable", False),
        ("Electric stove", "Anafe eléctrico", False), ("Gas stove", "Cocina a gas", False),
        ("Barbecue utensils", "Utensilios de parrilla", False), ("Children’s dinnerware", "Vajilla para chicos", False), ("Kitchenette", "Kitchenette", False),
    ],
    "Baño": [
        ("Hot water", "Agua caliente", True), ("Hair dryer", "Secador de pelo", True), ("Bidet", "Bidet", True),
        ("Bathtub", "Bañera", True), ("Shampoo", "Shampoo", True), ("Body soap", "Jabón corporal", False),
        ("Conditioner", "Acondicionador", False), ("Shower gel", "Gel de ducha", False),
        ("Outdoor shower", "Ducha exterior", False),
    ],
    "Dormitorio y lavado": [
        ("Bed linens", "Ropa de cama", True), ("Hangers", "Perchas", True),
        ("Clothing storage", "Placard o guardarropa", True),
        ("Essentials", "Básicos (toallas, sábanas, jabón y papel higiénico)", True),
        ("Room-darkening shades", "Cortinas blackout", True), ("Iron", "Plancha", True),
        ("Extra pillows and blankets", "Almohadas y mantas extra", True),
        ("Cleaning products", "Productos de limpieza", True), ("Washer", "Lavarropas", False),
        ("Drying rack for clothing", "Tendedero", False), ("Free washer", "Lavarropas sin cargo", False),
        ("Paid washer", "Lavarropas pago", False), ("Dryer", "Secarropas", False),
        ("Free dryer", "Secarropas sin cargo", False), ("Paid dryer", "Secarropas pago", False),
        ("Laundromat nearby", "Lavandería cerca", False), ("Lock on bedroom door", "Cerradura en el dormitorio", False),
        ("Mosquito net", "Mosquitero", False), ("Safe", "Caja fuerte", False),
    ],
    "Clima": [
        ("Air conditioning", "Aire acondicionado", True), ("AC", "Equipo de aire (AC)", False),
        ("Window AC unit", "Aire de ventana", False), ("Heating", "Calefacción", True),
        ("Radiant heating", "Losa radiante", False), ("Central heating", "Calefacción central", False),
        ("Portable heater", "Estufa portátil", False), ("Ceiling fan", "Ventilador de techo", False),
        ("Portable fans", "Ventiladores portátiles", False), ("Indoor fireplace", "Hogar a leña", False),
    ],
    "Internet, trabajo y entretenimiento": [
        ("Wifi", "Wifi", True), ("TV", "TV", True), ("Dedicated workspace", "Espacio de trabajo", True),
        ("Fast wifi", "Wifi rápido", False), ("Ethernet connection", "Conexión por cable (Ethernet)", False),
        ("Pocket wifi", "Wifi portátil", False), ("HDTV", "TV HD", False),
        ("Books and reading material", "Libros", False), ("Board games", "Juegos de mesa", False),
        ("Sound system", "Equipo de sonido", False),
    ],
    "Edificio y estacionamiento": [
        ("Elevator", "Ascensor", True), ("Building staff", "Personal del edificio", False),
        ("Doorman", "Portero", False), ("Private entrance", "Entrada privada", False),
        ("Single level home", "Una sola planta", False), ("Free parking on premises", "Cochera sin cargo", False),
        ("Paid parking on premises", "Cochera paga", False),
        ("Free street parking", "Estacionamiento sin cargo en la calle", False),
        ("Paid street parking off premises", "Estacionamiento pago en la calle", False),
        ("Paid parking off premises", "Estacionamiento pago cerca", False),
        ("Paid parking lot off premises", "Playa de estacionamiento paga cerca", False),
        ("Pool", "Pileta", False), ("Shared pool", "Pileta compartida", False),
        ("Shared outdoor pool", "Pileta compartida al aire libre", False), ("Hot tub", "Jacuzzi", False),
        ("Shared sauna", "Sauna compartido", False), ("Sauna", "Sauna", False),
        ("Shared hot tub", "Jacuzzi compartido", False), ("Gym", "Gimnasio", False),
        ("Shared gym", "Gimnasio compartido", False), ("Exercise equipment", "Equipo de ejercicio", False),
    ],
    "Exterior y vistas": [
        ("Private patio or balcony", "Balcón o patio privado", False), ("Patio or balcony", "Balcón o patio", False),
        ("Shared patio or balcony", "Balcón o patio compartido", False),
        ("Outdoor dining area", "Comedor al aire libre", False), ("Outdoor furniture", "Muebles de exterior", False),
        ("BBQ grill", "Parrilla", False), ("Shared BBQ grill", "Parrilla compartida", False),
        ("Backyard", "Jardín", False), ("Private backyard", "Jardín privado", False),
        ("Shared backyard", "Jardín compartido", False), ("Sun loungers", "Reposeras", False),
        ("City skyline view", "Vista a la ciudad", False), ("Courtyard view", "Vista a patio interno", False),
        ("Private BBQ grill", "Parrilla privada", False), ("Hammock", "Hamaca paraguaya", False),
        ("Garden view", "Vista a jardín", False), ("Park view", "Vista a parque", False),
        ("River view", "Vista al río", False),
    ],
    "Seguridad y acceso": [
        ("Fire extinguisher", "Matafuegos", False), ("First aid kit", "Botiquín", False),
        ("Smoke alarm", "Detector de humo", False), ("Carbon monoxide alarm", "Detector de monóxido de carbono", False),
        ("Exterior security cameras on property", "Cámaras de seguridad exteriores", False),
        ("Window guards", "Protecciones en ventanas", False), ("Lockbox", "Caja de llaves", False),
        ("Keypad", "Cerradura con código", False), ("Smart lock", "Cerradura inteligente", False),
    ],
    "Familia": [
        ("Crib", "Cuna", False), ("Pack ’n play/Travel crib", "Cuna de viaje", False),
        ("High chair", "Silla alta", False), ("Babysitter recommendations", "Recomendaciones de niñera", False),
        ("Baby bath", "Bañera para bebé", False), ("Children’s books and toys", "Libros y juguetes infantiles", False),
    ],
    "Servicios y reglas": [
        ("Long term stays allowed", "Se permiten estadías largas", True), ("Self check-in", "Ingreso autónomo", False),
        ("Host greets you", "El anfitrión te recibe", False), ("Luggage dropoff allowed", "Se puede dejar equipaje", False),
        ("Pets allowed", "Se aceptan mascotas", False), ("Smoking allowed", "Se permite fumar", False),
        ("Cleaning available during stay", "Limpieza durante la estadía", False),
        ("Housekeeping", "Servicio de mucama", False), ("Breakfast", "Desayuno", False), ("Bikes", "Bicicletas", False),
    ],
}



def columnas_que_activa(nombre_en):
    """Columnas has_* del modelo que activa esta comodidad, según las reglas del pipeline."""
    return [col for col, (rx, excl, _) in REGLAS_MODELO.items()
            if re.search(rx, nombre_en, re.IGNORECASE) and not (excl and re.search(excl, nombre_en, re.IGNORECASE))]


# Desplegables: solo las que no activan ninguna columna del modelo (esas ya están en MODELO).
CATEGORIAS = {cat: [it for it in items if not columnas_que_activa(it[0])] for cat, items in CATEGORIAS.items()}
TODAS = [(cat, en, es, d) for cat, items in CATEGORIAS.items() for en, es, d in items]


def ayuda(nombre_en):
    return (f"En Airbnb figura como «{nombre_en}». No es una columna del modelo: suma 1 a la cantidad de comodidades, "
            "que sí lo es.")
