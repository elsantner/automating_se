importCpg("cpg.bin")

val calls = cpg.call.lineNumber.toJson
val controlStructure = cpg.controlStructure.lineNumber.toJson
val assignment = cpg.assignment.lineNumber.toJson
val local = cpg.local.lineNumber.toJson

val concat = s""""$calls","$controlStructure","$assignment","$local"""" 
concat |>> "./out/out.csv"