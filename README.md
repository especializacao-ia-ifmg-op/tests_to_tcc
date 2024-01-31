## Multivariate Reference Evapotranspiration Forecasting with CNN

### Representation

*_1 variável:_*
- dataset = [eto0, eto1, eto2, eto3, eto4, eto5, ...eto6939]
- X_train = [[eto0, eto1, eto2, eto3], [eto1, eto2, eto3, eto4], ..., [eto6935, eto6936, eto6937, eto6938]]
- y_train = [[eto4], [eto5], [eto6], [eto7], ..., [eto6939]]

*_2 variáveis:_*
- dataset = [[eto0, eto1, eto2, eto3, eto4, eto5, ...eto6939], [rs0, rs1, rs2, rs3, rs4, rs5, ...rs6939]]
- X_train = [[eto0, rs0, eto1, rs1, eto2, rs2, eto3, rs3], [eto1, rs1, eto2, rs2, eto3, rs3, eto4, rs4], ..., [eto6935, rs6935, eto6936, rs6936, eto6937, rs6937, eto6938, rs6938]]
- y_train = [[eto4], [eto5], [eto6], [eto7], ..., [eto6939]]

*_3 variáveis:_*
- dataset = [[eto0, eto1, eto2, eto3, eto4, eto5, ...eto6939], [rs0, rs1, rs2, rs3, rs4, rs5, ...rs6939], [tmax0, tmax1, tmax2, tmax3, tmax4, tmax5, ...tmax6939]]
- X_train = [[eto0, rs0, tmax0, eto1, rs1, tmax1, eto2, rs2, tmax2, eto3, rs3, tmax3], [eto1, rs1, tmax1, eto2, rs2, tmax1, eto3, rs3, tmax3, eto4, rs4, tmax4], ..., [eto6935, rs6935, tmax6935, eto6936, rs6936, tmax6936, eto6937, rs6937, tmax6937, eto6938, rs6938, tmax6938]]
- y_train = [[eto4], [eto5], [eto6], [eto7], ..., [eto6939]]
