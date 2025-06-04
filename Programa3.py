#Hecho por:
# Rodríguez Felipe Adrián Eduardo
# Velazquéz Marin César Alexis
import os
import numpy  as np 
import math
import matplotlib.pyplot as plt
'''Ordenamos los puntos mediante de la función sorted y la función lambda,
que nos permite ordenar los puntos por el primer elemento'''
def ordenar_puntos(puntos):
    return sorted(puntos, key=lambda p: p[0])

'''Hacemos la interpolación de Lagrange con los puntos ordenados'''
def Interpolacion_Lagrange(puntos, xi, grado):
    puntos_ordenados = ordenar_puntos(puntos)
    n = len(puntos_ordenados)
    
    '''Encontramos posición inicial para el grado solicitado'''
    inicio = 0
    while inicio + grado < n and puntos_ordenados[inicio + grado][0] < xi:
        inicio += 1
    
    seleccion = puntos_ordenados[inicio:inicio+grado+1]
    '''Aqui se aplica el metodo de Lagrange'''
    resultado = 0.0
    for i in range(len(seleccion)):
        term = seleccion[i][1]
        for j in range(len(seleccion)):
            if i != j:
                term *= (xi - seleccion[j][0]) / (seleccion[i][0] - seleccion[j][0])
        resultado += term
    return resultado
def leer_datos():
    while True:
        try:
            n = int(input("¿Cuántos puntos tiene la tabla? (mínimo 2): "))
            if n >= 2: break
            print("Error: Debe ingresar al menos 2 puntos")
        except: 
            print("Ingrese un número válido")
    
    datos = []
    print("\nIngrese los puntos (x y):")
    for i in range(n):
        while True:
            try:
                x, y = map(float, input(f"Punto {i+1}: ").split())
                datos.append([x, y])
                break
            except: 
                print("Error: Ingrese dos números separados por espacio")
    return datos

# Imprime la tabla de datos (índice, x, y)
def imprimir_tabla(datos):
    print("|  i  |    x    |    y    |")
    for i, (x, y) in enumerate(datos):
        print(f"| {i:3} | {x:7.3f} | {y:7.3f} |")

# Permite modificar un punto existente en la tabla
def modificar_dato(datos):
    idx = int(input("Índice del punto a modificar: "))  # Índice a modificar
    x = float(input("Nuevo valor de x: "))
    y = float(input("Nuevo valor de y: "))
    datos[idx] = [x, y]  # Reemplaza el punto con los nuevos valores
    return datos

# Construye la matriz normal (A) y el vector b para el sistema lineal Ax=b
def construir_matriz_normal(datos, grado):
    N = len(datos)
    A = np.zeros((grado + 1, grado + 1))  # Matriz de coeficientes
    b = np.zeros(grado + 1)  # Vector del lado derecho
    for i in range(grado + 1):
        for j in range(grado + 1):
            A[i][j] = sum(x[0] ** (i + j) for x in datos)  # Suma de potencias de x
        b[i] = sum((x[0] ** i) * x[1] for x in datos)  # Suma de x^i * y
    return A, b

# Calcula los coeficientes del polinomio de ajuste resolviendo el sistema Ax=b
def Ajuste_polinomial(datos, grado):
    A, b = construir_matriz_normal(datos, grado)
    print("\nMatriz del sistema normal A:")
    print(A)
    print("\nVector del lado derecho b:")
    print(b)
    coeficientes = np.linalg.solve(A, b)  # Resuelve el sistema
    print("\nCoeficientes del polinomio:")
    for i, a in enumerate(coeficientes):
        print(f"a[{i}] = {a}")
    return coeficientes

# Evalúa el polinomio para un valor x
def evaluar_polinomio(coefs, x):
    return sum(coefs[i] * x ** i for i in range(len(coefs)))

# Calcula la suma de los errores al cuadrado para un conjunto de datos
def error_cuadrado_total(datos, coefs, f_eval):
    return sum((f_eval(coefs, x[0]) - x[1]) ** 2 for x in datos)

# Genera la gráfica del ajuste y de los datos
def graficar_ajuste(datos, coefs, f_eval, titulo):
    x_vals = [x[0] for x in datos]
    y_vals = [x[1] for x in datos]
    x_line = np.linspace(min(x_vals), max(x_vals), 100)  # Línea para la curva
    y_line = [f_eval(coefs, xi) for xi in x_line]
    plt.scatter(x_vals, y_vals, color='blue', label='Datos')
    plt.plot(x_line, y_line, color='red', label='Ajuste')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(titulo)
    plt.show()

# Ajuste para modelo exponencial: y = C e^(A x)
def Ajuste_exponencial(datos):
    X = np.array([x[0] for x in datos])  # x originales
    Y = np.log([x[1] for x in datos])  # y transformados ln(y)
    N = len(X)
    sumX = np.sum(X)
    sumY = np.sum(Y)
    sumX2 = np.sum(X**2)
    sumXY = np.sum(X*Y)
    A = (N*sumXY - sumX*sumY) / (N*sumX2 - sumX**2)  # Pendiente
    B = (sumY - A*sumX) / N  # Intersección
    C = np.exp(B)  # Se obtiene C (ya no en logaritmos)
    print(f"\nModelo: y = {C} e^({A} x)")
    def modelo_exp(coefs, x): return coefs[0] * np.exp(coefs[1]*x)
    coefs = [C, A]
    error = error_cuadrado_total(datos, coefs, modelo_exp)
    print("Suma de los errores al cuadrado:", error)
    graficar_ajuste(datos, coefs, modelo_exp, "Ajuste Exponencial")

# Ajuste para modelo potencial: y = C x^A
def Ajuste_potencial(datos):
    X = np.log([x[0] for x in datos])
    Y = np.log([x[1] for x in datos])
    N = len(X)
    sumX = np.sum(X)
    sumY = np.sum(Y)
    sumX2 = np.sum(X**2)
    sumXY = np.sum(X*Y)
    A = (N*sumXY - sumX*sumY) / (N*sumX2 - sumX**2)  # Exponente
    B = (sumY - A*sumX) / N
    C = np.exp(B)
    print(f"\nModelo: y = {C} x^{A}")
    def modelo_pot(coefs, x): return coefs[0] * x**coefs[1]
    coefs = [C, A]
    error = error_cuadrado_total(datos, coefs, modelo_pot)
    print("Suma de los errores al cuadrado:", error)
    graficar_ajuste(datos, coefs, modelo_pot, "Ajuste Potencial")

# Menú principal para elegir el tipo de ajuste y repetir con los mismos o nuevos datos
def minimos_cuadrados():
    datos = leer_datos()
    imprimir_tabla(datos)
    while True:
        correcto = input("¿Los datos son correctos? (s/n): ")
        if correcto.lower() == 's':
            break
        else:
            datos = modificar_dato(datos)
            imprimir_tabla(datos)
    while True:
        print("\n1. Ajuste Polinomial\n2. Ajuste Exponencial\n3. Ajuste Potencial\n4. Salir")
        op = input("Elija una opción: ")
        if op == '1':
            grado = int(input("Ingrese el grado del polinomio (tiene que ser menor al número de puntos): "))
            if grado < 1 or grado >= len(datos):
                print("Grado inválido. Debe ser al menos 1 y menor que el número de puntos.")
                continue
            coefs = Ajuste_polinomial(datos, grado)
            error = error_cuadrado_total(datos, coefs, evaluar_polinomio)
            print("\nSuma de los errores al cuadrado:", error)
            graficar_ajuste(datos, coefs, evaluar_polinomio, f"Ajuste Polinomial grado {grado}")
        elif op == '2':
            Ajuste_exponencial(datos)
        elif op == '3':
            Ajuste_potencial(datos)
        elif op == '4':
            break
        else:
            print("Opción no válida.")
        otra = input("\n¿Desea realizar otro ajuste con la misma tabla? (s/n): ")
        if otra.lower() != 's':
            break
    nueva_tabla = input("\n¿Desea realizar otro ajuste con otra tabla? (s/n): ")
    if nueva_tabla.lower() == 's':
        minimos_cuadrados()
#Menu principal
#Agregamos la tercera opción de Mínimos Cuadrados para el paquete final
print("Desarrolladores:\n- Rodríguez Felipe Adrián Eduardo\n- Velazquéz Marin César Alexis")
while True:
    print("\nMenu Principal") 
    print("2. Interpolación Polinomial de Lagrange")
    print("3. Minimos Cuadrados")
    print("4. Salir\n")
    opcion = input("Seleccione una opción: ")
    if opcion == '2':
        #Lectura del numero de puntos
        while True:
            try:
                os.system('cls')
                print("Interpolación Polinomial de Lagrange")
                n = int(input("\nNúmero de puntos (mínimo 2): "))
                if n >= 2: break
                print("Error: Debe ingresar al menos 2 puntos")
            except: print("Ingrese un número válido")
        
        #Lectura de puntos(tabla)
        while True:
            puntos = []
            print("\nIngrese los puntos (x y):")
            for i in range(n):
                while True:
                    try:
                        x, y = map(float, input(f"Punto {i+1}: ").split())
                        puntos.append([x, y])
                        break
                    except: print("Error: Ingrese dos números separados por espacio")
        
            #Mostramos los datos
            print("\nDatos ingresados:")
            for i, (x, y) in enumerate(puntos):
                print(f"Punto {i+1}: ({x:.4f}, {y:.4f})")
        
            if input("\n¿Son correctos? (s/n): ").lower() == 's': break
        
        #Ordenamos los puntos, con la función ya definida
        puntos = ordenar_puntos(puntos)
       
        #Bucle de interpolación
        while True:
            #Lectura de x a interpolar
            while True:
                try:
                    xi = float(input("\nValor x a interpolar: "))
                    if puntos[0][0] <= xi <= puntos[-1][0]: break
                    print(f"Debe estar entre [{puntos[0][0]:.4f}, {puntos[-1][0]:.4f}]")
                except: print("Número válido")
            
            #Lectura del grado
            while True:
                try:
                    grado = int(input(f"Grado (1-{n-1}): "))
                    if 1 <= grado <= n-1: break
                    print(f"Grado debe ser 1-{n-1}")
                except: print("Número válido")
            
            #Calculo y muestra del resultado
            resultado = Interpolacion_Lagrange(puntos, xi, grado)
            print(f"\nResultado: f({xi:.4f}) = {resultado:.4f}")
            
            if input("\n¿Interpolar otro punto? (s/n): ").lower() != 's': break
        
        if input("\n¿Realizar otra interpolación con nuevos puntos? (s/n): ").lower() != 's':
            print("\nRegresando al menú principal...") #Se limpia mas rapido de lo que se llega a ver xd
        
        os.system('cls') 
    elif opcion == '3':
        print("\nMínimos cuadrados (polinomial y no polinomial)")
        #Se coloco de esta manera para que no se repitiera el código de la función y
        #no se confundiera con el otro menu, propio de minimos cuadrados
        minimos_cuadrados()    
    
    elif opcion == '4':
        print("\nPrograma terminado.")
        break
    
    else:
        print("\nOpción no válida. Intente nuevamente.")        

