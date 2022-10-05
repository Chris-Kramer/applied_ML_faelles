# applied_ML_Faelles
## Guide til git

### Krav til at git  
Før  at git skal benyttes skal i have en terminal, der har git. Jeg mener at visual studio code allerede har git i terminalen.  
Ellers git bash til windows benyttes (min personlige favorit) kan hentes her https://gitforwindows.org/ 
Hvis I ikke vil benytte terminalen kan source tree benyttes (Andreas ved mere om dette) kan hentes her: https://www.sourcetreeapp.com/  


### Oprette repository fra bunden  
```bash  
git clone https://github.com/Chris-Kramer/applied_ML_faelles.git  
```

### Hente seneste version af repository
Sørg for at du befinder dig i folderen "applied ML_faelles  

```bash  
git pull origin main  
```

### Tilføj nye redigeringer

First skriv  
```bash  
git add -A  
```

Herefter  
```bash  
git commit -m "skriv en besked her"  
```

Til sidst  
```bash  
git push origin main  
```
### BEMÆRK  
Start gerne med at hente seneste version (pull) inden i laver redigeringer :D  

# Source Tree
HVis man vil benytte source tree skal man hente programmet og tilføje ssh-nøglen